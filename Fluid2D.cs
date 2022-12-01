using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Fluid2D : MonoBehaviour
{
    [Header("simulation")]
    public Vector2Int gridCount = new Vector2Int(256, 256);
    public Vector2 cellSize = new Vector2(.01f,.01f);
    public float timeStep=.01f;
    public float viscosity = 0f;
    public float vorticity_eps = .01f;
    public int pressure_iteration = 40;//40-80
    public int viscosity_iteration = 20;//20-50
    public bool useMacCormack = true;
    public bool resetPressure = true;

    [Header("initial values")]
    public float initial_velocity = 10f;

    [Header("display")]
    public float update_interval = .01f;
    public float color_scale = 15f;
    public float display_size = 10;
    public DisplayChannel display_channel;

    RenderTexture velocity;
    RenderTexture velocity_new;
    RenderTexture velocity_divergence;
    RenderTexture pressure;
    RenderTexture pressure_new;
    RenderTexture vorticity;
    [System.Serializable]public enum DisplayChannel { Velocity,Divergence,Pressure, Vorticity};


    public ComputeShader shader;

    private void Start()
    {
        Init();
        SetInitialCondition();
        StartCoroutine(MainLoop());
    }
    IEnumerator MainLoop()
    {
        while (true)
        {
            Step();
            transform.localScale = new Vector3(gridCount.x, gridCount.y, 1) / Mathf.Max(gridCount.x, gridCount.y) * display_size;
            var mat = GetComponent<MeshRenderer>().material;
            var texs = new Texture[] { velocity, velocity_divergence, pressure, vorticity};
            mat.SetTexture("_MainTex", texs[((int)display_channel)]);
            mat.SetFloat("color_scale", color_scale);

            if (update_interval > 0)
                yield return new WaitForSeconds(update_interval);
            else
                yield return null;
        }
    }

    public void Init()
    {
        var desc = new RenderTextureDescriptor(gridCount.x, gridCount.y);
        desc.enableRandomWrite = true;

        desc.colorFormat = RenderTextureFormat.RGFloat;//float2
        velocity = new RenderTexture(desc);
        velocity_new = new RenderTexture(desc);

        desc.colorFormat = RenderTextureFormat.RFloat;//float
        velocity_divergence = new RenderTexture(desc);
        pressure = new RenderTexture(desc);
        pressure_new = new RenderTexture(desc);
        vorticity = new RenderTexture(desc);
    }
    public void SetInitialCondition()
    {
        int kid; Vector3Int groupCount;
        // Advect Diffuse Vorticity AddForce RemovePressure
        shader.SetInts("gridCount", new int[] { gridCount.x, gridCount.y });
        shader.SetFloats("cellSize", new float[] { cellSize.x, cellSize.y });
        shader.SetFloat("timeStep", timeStep);
        {
            kid = shader.FindKernel("InitCondition");
            shader.SetFloat("initial_velocity", initial_velocity);
            shader.SetTexture(kid, "output_x", velocity);
            groupCount = CalcGroupCount(kid);
            shader.Dispatch(kid, groupCount.x, groupCount.y, groupCount.z);
        }

    }
    public void Step()
    {
        shader.SetInts("gridCount", new int[] { gridCount.x, gridCount.y });
        shader.SetFloats("cellSize", new float[] { cellSize.x, cellSize.y });
        shader.SetFloat("timeStep", timeStep);

        //Advect Velocity Field
        RunKernel2D("AdvectionFloat2", "input_vector", velocity, "input_x", velocity, "output_x", velocity_new);
        Swap(ref velocity_new, ref velocity);
        if (useMacCormack)
        {
            RunKernel2D("AdvectionRefine", "input_x", velocity, "output_x", velocity_new);
            Swap(ref velocity_new, ref velocity);
        }

        //Diffuse Velocity using Jacobi Method
        if (viscosity > 0)
        {
            Vector2 L= -viscosity*timeStep* new Vector2(1.0f / (cellSize.x * cellSize.x), 1.0f / (cellSize.y * cellSize.y));
            float Sigma = 1 - 2 * (L.x + L.y);
            shader.SetFloat("jac_inv_diagCoeff", 1 / Sigma);
            shader.SetVector("jac_neighborCoeff_mod", -L / Sigma);
            for (int i = 0; i < viscosity_iteration; ++i)
            {
                RunKernel2D("JacobiIterationDirichletFloat2", "input_x", velocity, "output_x", velocity_new, "input_b", velocity);
                Swap(ref velocity_new, ref velocity);
            }
        }

        //ω=▽×u
        RunKernel2D("Curl", "input_vector", velocity, "output_x", vorticity);
        //u=u+eps dt dx normalize(▽|ω|)×ω
        shader.SetFloat("vorticity_eps", vorticity_eps);
        RunKernel2D("AddVorticity", "input_scalar", vorticity, "output_x", velocity);

        RunKernel2D("AddForce", "output_x", velocity);

        //Remove Pressure
        //b=▽@w / Δt
        RunKernel2D("DivergenceMod", "input_vector", velocity, "output_x", velocity_divergence);
        //▽@▽p=b 
        {
            if (resetPressure)
            {
                shader.SetVector("fill_value", Vector4.zero);
                RunKernel2D("FillTextureFloat", "output_x", pressure);
            }

            Vector2 L = new Vector2(1.0f / (cellSize.x * cellSize.x), 1.0f / (cellSize.y * cellSize.y));
            float Sigma = -2 * (L.x + L.y);
            shader.SetFloat("jac_inv_diagCoeff", 1/ Sigma);
            shader.SetVector("jac_neighborCoeff_mod", -L/Sigma);
            for (int i = 0; i < pressure_iteration; ++i)
            {
                RunKernel2D("JacobiIterationNeumannFloat", "input_x", pressure, "output_x", pressure_new, "input_b", velocity_divergence);
                Swap(ref pressure_new, ref pressure);
            }
        }
        //u=w-▽p Δt
        RunKernel2D("MinusGradientMod", "input_scalar", pressure, "output_x", velocity);

    }
    /*
    public void _Step()
    {
        int kid; Vector3Int groupCount;
        // Advect Diffuse AddForce RemovePressure
        shader.SetInts("gridCount", new int[] { gridCount.x, gridCount.y });
        shader.SetFloat("cellSize", cellSize);
        shader.SetFloat("timeStep", timeStep);

        //Advect Velocity Field
        {
            kid = shader.FindKernel("AdvectionFloat2");
            shader.SetTexture(kid, "input_vector", velocity);
            shader.SetTexture(kid, "input_x", velocity);
            shader.SetTexture(kid, "output_x", velocity_new);
            groupCount = CalcGroupCount(kid);
            shader.Dispatch(kid, groupCount.x, groupCount.y, groupCount.z);
            Swap(ref velocity_new, ref velocity);
        }
        if (useMacCormack)
        {
            kid = shader.FindKernel("AdvectionRefine");
            shader.SetTexture(kid, "input_x", velocity);
            shader.SetTexture(kid, "output_x", velocity_new);
            groupCount = CalcGroupCount(kid);
            shader.Dispatch(kid, groupCount.x, groupCount.y, groupCount.z);
            Swap(ref velocity_new, ref velocity);
        }
        //Diffuse Velocity using Jacobi Method
        if (viscosity > 0)
            for (int i = 0; i < viscosity_iteration; ++i)
            {
                kid = shader.FindKernel("JacobiIterationFloat2");
                shader.SetTexture(kid, "input_x", velocity);
                shader.SetTexture(kid, "output_x", velocity_new);
                shader.SetTexture(kid, "input_b", velocity);
                float tmp = cellSize * cellSize + 2 * dim * viscosity * timeStep;
                float alpha_inv_beta = cellSize * cellSize / tmp;
                float inv_beta = viscosity * timeStep / tmp;
                shader.SetFloat("jac_alpha_inv_beta", alpha_inv_beta);
                shader.SetFloat("jac_inv_beta", inv_beta);
                groupCount = CalcGroupCount(kid);
                shader.Dispatch(kid, groupCount.x, groupCount.y, groupCount.z);
                Swap(ref velocity_new, ref velocity);
            }

        //Vorticity Constraint
        //ω=▽×u
        {
            kid = shader.FindKernel("Curl");
            shader.SetTexture(kid, "input_vector", velocity);
            shader.SetTexture(kid, "output_x", vorticity);
            groupCount = CalcGroupCount(kid);
            shader.Dispatch(kid, groupCount.x, groupCount.y, groupCount.z);
        }
        //u=u+eps dt dx normalize(▽|ω|)×ω
        {
            kid = shader.FindKernel("AddVorticity");
            shader.SetFloat("vorticity_eps", vorticity_eps);
            shader.SetTexture(kid, "input_scalar", vorticity);
            shader.SetTexture(kid, "output_x", velocity);
            groupCount = CalcGroupCount(kid);
            shader.Dispatch(kid, groupCount.x, groupCount.y, groupCount.z);
        }

        //Remove Pressure
        //b=▽@w
        {
            kid = shader.FindKernel("Divergence");
            shader.SetTexture(kid, "input_vector", velocity);
            shader.SetTexture(kid, "output_x", velocity_divergence);
            groupCount = CalcGroupCount(kid);
            shader.Dispatch(kid, groupCount.x, groupCount.y, groupCount.z);
        }
        //▽@▽p=b
        for (int i=0;i<pressure_iteration;++i)
        {
            {
                kid = shader.FindKernel("JacobiIterationFloat");
                shader.SetTexture(kid, "input_x", pressure);
                shader.SetTexture(kid, "output_x", pressure_new);
                shader.SetTexture(kid, "input_b", velocity_divergence);

                float alpha_inv_beta = -cellSize * cellSize / (2 * dim);
                float inv_beta = 1f / (2 * dim);
                shader.SetFloat("jac_alpha_inv_beta", alpha_inv_beta);
                shader.SetFloat("jac_inv_beta", inv_beta);
                groupCount = CalcGroupCount(kid);
                shader.Dispatch(kid, groupCount.x, groupCount.y, groupCount.z);
                Swap(ref pressure_new, ref pressure);
            }
            {
                kid = shader.FindKernel("UpdateNeumannBoundaryFloat");
                shader.SetTexture(kid, "output_x", pressure);
                groupCount = CalcGroupCount(kid, isBoundary: true);
                shader.Dispatch(kid, groupCount.x, groupCount.y, groupCount.z);
            }
        }
        //u=w-▽p
        {
            kid = shader.FindKernel("MinusGradient");
            shader.SetTexture(kid, "input_scalar", pressure);
            shader.SetTexture(kid, "output_x", velocity);
            groupCount = CalcGroupCount(kid);
            shader.Dispatch(kid, groupCount.x, groupCount.y, groupCount.z);
        }
    }
    */

    void RunKernel2D(string kernelName, params object[] textures)
    {
        int kid = shader.FindKernel(kernelName);
        for (int i = 0; i < textures.Length; i += 2)
            shader.SetTexture(kid, (string)textures[i], (Texture)textures[i + 1]);
        Vector3Int groupCount = CalcGroupCount(kid);
        shader.Dispatch(kid, groupCount.x, groupCount.y, groupCount.z);

    }
    Vector3Int CalcGroupCount(int kid,bool isBoundary=false)
    {
        shader.GetKernelThreadGroupSizes(kid, out uint sx, out uint sy, out uint sz);
        if (isBoundary)
            return new Vector3Int(Mathf.CeilToInt((float)gridCount.x * gridCount.y / sx), 1, 1);
        else
            return new Vector3Int(
                    Mathf.CeilToInt((float)gridCount.x / sx),
                    Mathf.CeilToInt((float)gridCount.y / sy),
                    1
                );
    }
    void Swap(ref RenderTexture a, ref RenderTexture b)
    {
        var tmp = a; a = b; b = tmp;
    }

}
