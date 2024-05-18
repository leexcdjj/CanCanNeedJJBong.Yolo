﻿using System.Collections.Concurrent;
using CanCanNeedJJBong.Yolo.Core.Basic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace CanCanNeedJJBong.Yolo.Core.TaskModelStrategy.YoloV5;

/// <summary>
/// YoloV5 detect模式(检测)
/// </summary>
public class DetectInferenceStrategyV5 : ITaskModelInferenceStrategy
{
    public List<YoloData> ExecuteTask(YoloConfig yolo,IReadOnlyCollection<NamedOnnxValue> container, float confidenceDegree, float iouThreshold, bool allIou)
    {
        throw new NotImplementedException("YoloV5 detect模式(检测)未实现");
    }
}