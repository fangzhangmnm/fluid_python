using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Cloud2D : MonoBehaviour
{
    [Header("simulation")]
    public Vector2Int gridCount = new Vector2Int(256, 256);
    public Vector2 cellSize = new Vector2(50,50);
    public float timeStep = 50;
    public float viscosity = 0f;
    public float vorticity_eps = .01f;
    public int pressure_iteration = 40;//40-80
    public int viscosity_iteration = 20;//20-50
    public bool useMacCormack = true;
    public bool resetPressure = true;

    [Header("thermodynamics")]
    public float bgSealevelTemperature = 288.15f;
    public float bgTemperatureGradient = -0.0065f;
    public float bgSealevelPressure = 101325f;
    public float bgPressureExponent = 5.2561f;
    public float heatCapacityKappa = 0.286f;
    public float referencePressure= 101325f;
    public float waterMolarMass= 28.96e-3f;
    public float airMolarMass= 18.02e-3f;
    public float vaporGamma= 1.33f;
    public float airGamma= 1.4f;
    public float gasConstantR= 8.314f;
    public float vaporLatentHeatPerMass= 2.5f;
    public float gravityStrength=+9.8f;


    [Header("display")]
    public float update_interval = .01f;
    public float color_scale = 1f;
    public Vector2 color_bias = new Vector2();
    public float display_size = 10;
    public DisplayChannel display_channel;

    private void Start()
    {
        Init();
        SetInitialCondition();
        Step();
        StartCoroutine(MainLoop());
    }

    IEnumerator MainLoop()
    {
        while (true)
        {
            Step();
            transform.localScale = new Vector3(gridCount.x, gridCount.y, 1) / Mathf.Max(gridCount.x, gridCount.y) * display_size;
            var mat = GetComponent<MeshRenderer>().material;
            var texs = new Texture[] {
                velocity,
                velocity_divergence,
                pressure,
                vorticity,
                potentialTemperature,
                vaporRatio,
                cloudRatio,
                rainRatio,
            };
            mat.SetTexture("_MainTex", texs[((int)display_channel)]);
            mat.SetFloat("color_scale", color_scale);
            mat.SetVector("color_bias", color_bias);

            if (update_interval > 0)
                yield return new WaitForSeconds(update_interval);
            else
                yield return null;
        }
    }

    void SetShaderVals()
    {
        shader.SetInts("gridCount", new int[] { gridCount.x, gridCount.y });
        shader.SetFloats("cellSize", new float[] { cellSize.x, cellSize.y });
        shader.SetFloat("timeStep", timeStep);

        shader.SetFloat("vorticity_eps", vorticity_eps);

        shader.SetFloat("bgSealevelTemperature", bgSealevelTemperature);
        shader.SetFloat("bgTemperatureGradient", bgTemperatureGradient);
        shader.SetFloat("bgSealevelPressure", bgSealevelPressure);
        shader.SetFloat("bgPressureExponent", bgPressureExponent);
        shader.SetFloat("heatCapacityKappa", heatCapacityKappa);
        shader.SetFloat("referencePressure", referencePressure);
        shader.SetFloat("waterMolarMass", waterMolarMass);
        shader.SetFloat("airMolarMass", airMolarMass);
        shader.SetFloat("vaporGamma", vaporGamma);
        shader.SetFloat("airGamma", airGamma);
        shader.SetFloat("gasConstantR", gasConstantR);
        shader.SetFloat("vaporLatentHeatPerMass", vaporLatentHeatPerMass);
        shader.SetFloat("gravityStrength", gravityStrength);
    }

    public void SetInitialCondition()
    {
        SetShaderVals();
        RunKernel2D("SetInitialValues",
            "bgMassDensity", bgMassDensity,
            "bgPressure", bgPressure,
            "bgTemperature", bgTemperature,
            "velocity", velocity,
            "potentialTemperature", potentialTemperature,
            "vaporRatio", vaporRatio,
            "cloudRatio", cloudRatio,
            "rainRatio", rainRatio);
    }

    public void Step()
    {
        SetShaderVals();

        //Advect Velocity Field
        RunKernel2D("AdvectionFloat2", "input_vector", velocity, "input_x", velocity, "output_x", vector_new);
        Swap(ref vector_new, ref velocity);
        if (useMacCormack)
        {
            RunKernel2D("AdvectionRefine", "input_x", velocity, "output_x", vector_new);
            Swap(ref vector_new, ref velocity);
        }

        //Diffuse Velocity using Jacobi Method
        if (viscosity > 0)
        {
            Vector2 L = -viscosity * timeStep * new Vector2(1.0f / (cellSize.x * cellSize.x), 1.0f / (cellSize.y * cellSize.y));
            float Sigma = 1 - 2 * (L.x + L.y);
            shader.SetFloat("jac_inv_diagCoeff", 1 / Sigma);
            shader.SetVector("jac_neighborCoeff_mod", -L / Sigma);
            for (int i = 0; i < viscosity_iteration; ++i)
            {
                RunKernel2D("JacobiIterationDirichletFloat2", "input_x", velocity, "output_x", vector_new, "input_b", velocity);
                Swap(ref vector_new, ref velocity);
            }
        }

        //¦Ø=¨Œ¡Áu
        RunKernel2D("Curl", "input_vector", velocity, "output_x", vorticity);
        //u=u+eps dt dx normalize(¨Œ|¦Ø|)¡Á¦Ø
        shader.SetFloat("vorticity_eps", vorticity_eps);
        RunKernel2D("AddVorticity", "input_scalar", vorticity, "output_x", velocity);

        // Add Buoyancy
        RunKernel2D("ApplyForce", "input_x", velocity, "output_x", velocity,
            "potentialTemperature", potentialTemperature,
            "bgPressure", bgPressure,
            "bgTemperature", bgTemperature,
            "vaporRatio", vaporRatio);
        
        //Remove Pressure
        //b=¨Œ@w
        RunKernel2D("DivergenceMod", "input_vector", velocity, "input_scalar_1D",bgMassDensity,"output_x", velocity_divergence);
        //¨Œ@¨Œp=b
        {
            if (resetPressure)
            {
                shader.SetVector("fill_value", Vector4.zero);
                RunKernel2D("FillTextureFloat", "output_x", pressure);
            }

            Vector2 L = new Vector2(1.0f / (cellSize.x * cellSize.x), 1.0f / (cellSize.y * cellSize.y));
            float Sigma = -2 * (L.x + L.y);
            shader.SetFloat("jac_inv_diagCoeff", 1 / Sigma);
            shader.SetVector("jac_neighborCoeff_mod", -L / Sigma);
            for (int i = 0; i < pressure_iteration; ++i)
            {
                RunKernel2D("JacobiIterationNeumannFloat", "input_x", pressure, "output_x", scalar_new, "input_b", velocity_divergence);
                Swap(ref scalar_new, ref pressure);
            }
        }
        //u=w-¨Œp
        RunKernel2D("MinusGradientMod", "input_scalar", pressure, "input_scalar_1D", bgMassDensity, "output_x", velocity);

        //Advect water&temp
        RunKernel2D("AdvectionFloat", "input_vector", velocity, "input_x", potentialTemperature, "output_x", scalar_new);
        Swap(ref scalar_new, ref potentialTemperature);
        RunKernel2D("AdvectionFloat", "input_vector", velocity, "input_x", vaporRatio, "output_x", scalar_new);
        Swap(ref scalar_new, ref vaporRatio);
        RunKernel2D("AdvectionFloat", "input_vector", velocity, "input_x", cloudRatio, "output_x", scalar_new);
        Swap(ref scalar_new, ref cloudRatio);
        RunKernel2D("AdvectionFloat", "input_vector", velocity, "input_x", rainRatio, "output_x", scalar_new);
        Swap(ref scalar_new, ref rainRatio);

        //Update water&temp
        RunKernel2D("UpdateCondensation",
            "vaporRatio", vaporRatio,
            "cloudRatio", cloudRatio,
            "rainRatio", rainRatio,
            "potentialTemperature", potentialTemperature,
            "bgPressure", bgPressure
            );
    }


    RenderTexture velocity;
    RenderTexture vector_new;
    RenderTexture velocity_divergence;
    RenderTexture pressure;
    RenderTexture scalar_new;
    RenderTexture vorticity;

    RenderTexture potentialTemperature;
    RenderTexture vaporRatio;
    RenderTexture cloudRatio;
    RenderTexture rainRatio;
    RenderTexture bgMassDensity;
    RenderTexture bgPressure;
    RenderTexture bgTemperature;

    [System.Serializable] public enum DisplayChannel
    {
        velocity,
        velocity_divergence,
        pressure,
        vorticity,
        potentialTemperature,
        vaporRatio,
        cloudRatio,
        rainRatio,
    };
    public void Init()
    {
        var desc = new RenderTextureDescriptor(gridCount.x, gridCount.y);
        desc.enableRandomWrite = true;

        desc.colorFormat = RenderTextureFormat.RGFloat;//float2
        velocity = new RenderTexture(desc);
        vector_new = new RenderTexture(desc);

        desc.colorFormat = RenderTextureFormat.RFloat;//float
        velocity_divergence = new RenderTexture(desc);
        pressure = new RenderTexture(desc);
        scalar_new = new RenderTexture(desc);
        vorticity = new RenderTexture(desc);
        potentialTemperature = new RenderTexture(desc);
        vaporRatio = new RenderTexture(desc);
        cloudRatio = new RenderTexture(desc);
        rainRatio = new RenderTexture(desc);


        desc = new RenderTextureDescriptor(1,gridCount.y);
        desc.enableRandomWrite = true;
        desc.colorFormat = RenderTextureFormat.RFloat;//float
        bgMassDensity = new RenderTexture(desc);
        bgPressure = new RenderTexture(desc);
        bgTemperature = new RenderTexture(desc);


    }

    public ComputeShader shader;

    void RunKernel2D(string kernelName, params object[] textures)
    {
        int kid = shader.FindKernel(kernelName);
        for (int i = 0; i < textures.Length; i += 2)
        {
            shader.SetTexture(kid, (string)textures[i], (Texture)textures[i + 1]);
        }
        Vector3Int groupCount = CalcGroupCount(kid);
        shader.Dispatch(kid, groupCount.x, groupCount.y, groupCount.z);

    }
    Vector3Int CalcGroupCount(int kid, bool isBoundary = false)
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
