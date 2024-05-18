using System.Collections.Concurrent;
using CanCanNeedJJBong.Yolo.Core.Basic;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace CanCanNeedJJBong.Yolo.Core.TaskModelStrategy.YoloV5;

/// <summary>
/// YoloV5 segment模式(分割)
/// </summary>
public class SegmentInferenceStrategyV5 : ITaskModelInferenceStrategy
{
    public List<YoloData> ExecuteTask(Tensor<float> data, float confidenceDegree, float iouThreshold, bool allIou, Yolo yolo)
    {
        throw new NotImplementedException("YoloV5 segment模式(分割)未实现");
    }
}