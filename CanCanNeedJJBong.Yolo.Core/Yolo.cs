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
}