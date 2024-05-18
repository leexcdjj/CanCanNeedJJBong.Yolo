using System.Collections.Concurrent;
using CanCanNeedJJBong.Yolo.Core.Basic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace CanCanNeedJJBong.Yolo.Core.TaskModelStrategy.YoloV5;

/// <summary>
/// YoloV5 segment模式(分割)
/// </summary>
public class SegmentInferenceStrategyV5 : ITaskModelInferenceStrategy
{
    public List<YoloData> ExecuteTask(Yolo yolo,IReadOnlyCollection<NamedOnnxValue> container, float confidenceDegree, float iouThreshold, bool allIou)
    {
        throw new NotImplementedException("YoloV5 segment模式(分割)未实现");
    }
}