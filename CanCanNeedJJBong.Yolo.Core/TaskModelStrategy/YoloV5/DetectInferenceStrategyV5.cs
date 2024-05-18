using System.Collections.Concurrent;
using CanCanNeedJJBong.Yolo.Core.Basic;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace CanCanNeedJJBong.Yolo.Core.TaskModelStrategy.YoloV5;

/// <summary>
/// YoloV5 detect模式(检测)
/// </summary>
public class DetectInferenceStrategyV5 : ITaskModelInferenceStrategy
{
    public List<YoloData> ExecuteTask(Tensor<float> data, float confidenceDegree, float iouThreshold, bool allIou, Yolo yolo)
    {
        throw new NotImplementedException("YoloV5 detect模式(检测)未实现");
    }
}