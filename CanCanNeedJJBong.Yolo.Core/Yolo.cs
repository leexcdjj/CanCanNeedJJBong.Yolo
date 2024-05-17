using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace CanCanNeedJJBong.Yolo.Core;

public class Yolo
{
    /// <summary>
    /// 模型会话
    /// </summary>
    public InferenceSession ModelSession { get; set; }

    /// <summary>
    /// 张量宽度
    /// </summary>
    public int TensorWidth { get; set; }

    /// <summary>
    /// 张量高度
    /// </summary>
    public int TensorHeight { get; set; }

    /// <summary>
    /// 模型输入名
    /// </summary>
    public string ModelInputName { get; set; }

    /// <summary>
    /// 模型输出名
    /// </summary>summary
    public string ModelOutputName { get; set; }

    /// <summary>
    /// 输入张量信息
    /// </summary>
    public int[] InputTensorInfo { get; set; }

    /// <summary>
    /// 输出张量信息
    /// </summary>
    public int[] OutputTensorInfo { get; set; }

    /// <summary>
    /// 输出张量信息2_分割
    /// </summary>
    public int[] OutputTensorInfo2_Segmentation { get; set; }

    /// <summary>
    /// 推理图片宽度
    /// </summary>
    public int InferenceImageWidth { get; set; }

    /// <summary>
    /// 推理图片高度
    /// </summary>
    public int InferenceImageHeight { get; set; }

    /// <summary>
    /// 输入张量
    /// </summary>
    public DenseTensor<float> InputTensor { get; set; }

    /// <summary>
    /// YOLO版本
    /// </summary>
    public int YoloVersion { get; set; }

    /// <summary>
    /// mask缩放比例W
    /// </summary>
    public float MaskScaleRatioW { get; set; } = 0;

    /// <summary>
    /// mask缩放比例H
    /// </summary>
    public float MaskScaleRatioH { get; set; } = 0;

    /// <summary>
    /// 模型版本
    /// </summary>
    public string ModelVersion { get; set; } = "";

    /// <summary>
    /// 任务类型
    /// </summary>
    public string TaskType { get; set; } = "";

    /// <summary>
    /// 语义分割宽度
    /// </summary>
    public int SemanticSegmentationWidth { get; set; } = 0;

    /// <summary>
    /// 动作宽度
    /// </summary>
    public int ActionWidth { get; set; } = 0;

    /// <summary>
    /// 缩放比例
    /// </summary>
    public float ScaleRatio { get; set; } = 1;

    /// <summary>
    /// 标签组
    /// </summary>
    public string[] LabelGroup { get; set; }

    /// <summary>
    /// 执行任务模式
    /// </summary>
    private int ExecutionTaskMode = 0;

    /// <summary>
    /// 任务模式
    /// </summary>
    public int TaskMode
    {
        get { return ExecutionTaskMode; }
        set
        {
            if (TaskType == "classify")
            {
                ExecutionTaskMode = 0;
            }
            //检测
            else if (TaskType == "detect")
            {
                ExecutionTaskMode = 1;
            }
            //分割
            else if (TaskType == "segment")
            {
                if (value == 1 || value == 2 || value == 3)
                {
                    ExecutionTaskMode = value;
                }
                else
                {
                    ExecutionTaskMode = 3;
                }
            }
            //动作
            else if (TaskType == "pose")
            {
                if (value == 1 || value == 4 || value == 5)
                {
                    ExecutionTaskMode = value;
                }
                else
                {
                    ExecutionTaskMode = 5;
                }
            }
            else if (TaskType == "obb")
            {
                if (value == 6)
                {
                    ExecutionTaskMode = value;
                }
                else
                {
                    ExecutionTaskMode = 6;
                }
            }
            else
            {
                //如果什么都数据都没有,默认是检测模型
                ExecutionTaskMode = 1;
            }
        }
    }


    /// <summary>
    /// 构造函数,目前支持yoloV5,yoloV6,yoloV8所转换的onnx模型,模型的详细信息可以推理网站https://netron.app/
    /// </summary>
    /// <param name="modelPath">模型路径:必须使用ONNX模型</param>      
    /// <param name="yoloVersion">yolo版本:如5,6,8为yolo版本号,默认为0自动检测,但可能会发生错误的判断,比如yolo6就需要指定，也可能是没有正确训练或进行过特殊调整导致的,如果错误判断,请手动指定</param>
    /// <param name="gpuIndex">gpuIndex:gpu索引</param>
    /// <param name="enableGpu">启用gpu:默认为false</param>
    public Yolo(string modelPath, int yoloVersion = 0, int gpuIndex = 0, bool enableGpu = false)
    {
        try
        {
            // 模型会话赋值
            if (enableGpu)
            {
                // 模式
                SessionOptions modo = new SessionOptions();
                modo.AppendExecutionProvider_DML(gpuIndex);
                ModelSession = new InferenceSession(modelPath, modo);
            }
            else
            {
                ModelSession = new InferenceSession(modelPath);
            }

            ModelInputName = ModelSession.InputNames.First();
            ModelOutputName = ModelSession.OutputNames.First();
            InputTensorInfo = ModelSession.InputMetadata[ModelInputName].Dimensions;
            OutputTensorInfo = ModelSession.OutputMetadata[ModelOutputName].Dimensions;
            
            // 模型信息
            var ModelInfo = ModelSession.ModelMetadata.CustomMetadataMap;
            
            if (ModelInfo.Keys.Contains("names"))
            {
                LabelGroup = SplitlabelSignature(ModelInfo["names"]);
            }
            else
            {
                LabelGroup = new string[0];
            }

            if (ModelInfo.Keys.Contains("version"))
            {
                ModelVersion = ModelInfo["version"];
            }

            if (ModelInfo.Keys.Contains("task"))
            {
                TaskType = ModelInfo["task"];
                if (TaskType == "segment")
                {
                    string 模型输出名2 = ModelSession.OutputNames[1];
                    OutputTensorInfo2_Segmentation = ModelSession.OutputMetadata[模型输出名2].Dimensions;
                    SemanticSegmentationWidth = OutputTensorInfo2_Segmentation[1];
                    MaskScaleRatioW = 1f * OutputTensorInfo2_Segmentation[3] / InputTensorInfo[3];
                    MaskScaleRatioH = 1f * OutputTensorInfo2_Segmentation[2] / InputTensorInfo[2];
                }
                else if (TaskType == "pose")
                {
                    if (OutputTensorInfo[1] > OutputTensorInfo[2])
                    {
                        ActionWidth = OutputTensorInfo[2] - 5;
                    }
                    else
                    {
                        ActionWidth = OutputTensorInfo[1] - 5;
                    }
                }
            }
            else
            {
                if (OutputTensorInfo.Length == 2)
                {
                    TaskType = "classify";
                }
                else if (OutputTensorInfo.Length == 3)
                {
                    if (ModelSession.OutputNames.Count == 1)
                    {
                        TaskType = "detect";
                    }
                    else if (ModelSession.OutputNames.Count == 2)
                    {
                        // 模型输出名2
                        string modelOutputNameTwo = ModelSession.OutputNames[1];
                        
                        OutputTensorInfo2_Segmentation = ModelSession.OutputMetadata[modelOutputNameTwo].Dimensions;
                        SemanticSegmentationWidth = OutputTensorInfo2_Segmentation[1];
                        MaskScaleRatioW = 1f * OutputTensorInfo2_Segmentation[3] / InputTensorInfo[3];
                        MaskScaleRatioH = 1f * OutputTensorInfo2_Segmentation[2] / InputTensorInfo[2];
                        TaskType = "segment";
                    }
                    else
                    {
                        // throw new Exception("暂不支持的模型");
                    }
                }
                else
                {
                    throw new Exception("暂不支持的模型");
                }
            }

            TaskMode = 0;
            YoloVersion = GetModelVersion(yoloVersion);
            TensorWidth = InputTensorInfo[3];
            TensorHeight = InputTensorInfo[2];
        }
        catch (Exception ex)
        {
            // todo: 日至组件
            // 错误打印日志
            Console.WriteLine($"错误：{ex.Message}");

            throw ex;
        }
    }
    
    /// <summary>
    /// 获取模型版本
    /// </summary>
    /// <param name="version"></param>
    /// <returns></returns>
    private int GetModelVersion(int version)
    {
        if (TaskType == "classify")
        {
            return 5;
        }

        if (version >= 8)
        {
            return 8;
        }
        else if (version < 8 && version >= 5)
        {
            return version;
        }

        if (ModelVersion != "")
        {
            return int.Parse(ModelVersion.Split('.')[0]);
        }

        int mid = OutputTensorInfo[1];
        int right = OutputTensorInfo[2];
        int size = mid < right ? mid : right;
        
        // 标签数量
        int lableCount = LabelGroup.Length;
        if (lableCount == size - 4 - SemanticSegmentationWidth)
        {
            return 8;
        }

        if (lableCount == 0 && mid < right)
        {
            return 8;
        }

        return 5;
    }
    
    /// <summary>
    /// 分割标签名
    /// </summary>
    /// <param name="name">名</param>
    /// <returns></returns>
    private string[] SplitlabelSignature(string name)
    {
        // 删除括号
        name = name.Replace("{", "").Replace("}", "");
        
        // 分割数组
        string[] SplitArr = name.Split(',');
        
        string[] result = new string[SplitArr.Length];
        
        for (int i = 0; i < SplitArr.Length; i++)
        {
            int start = SplitArr[i].IndexOf(':') + 3;
            int end = SplitArr[i].Length - 1;
            result[i] = SplitArr[i].Substring(start, end - start);
        }

        return result;
    }
    
}