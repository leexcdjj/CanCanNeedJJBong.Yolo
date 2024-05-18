using System.Collections.Concurrent;
using CanCanNeedJJBong.Yolo.Core.Basic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace CanCanNeedJJBong.Yolo.Core.TaskModelStrategy.YoloV6;

/// <summary>
/// YoloV6 segment模式(分割)
/// </summary>
public class SegmentInferenceStrategyV6 : ITaskModelInferenceStrategy
{
    public List<YoloData> ExecuteTask(YoloConfig yolo,IReadOnlyCollection<NamedOnnxValue> container, float confidenceDegree, float iouThreshold, bool allIou)
    {
        throw new NotImplementedException("YoloV6 segment模式(分割)未实现");
    }
}