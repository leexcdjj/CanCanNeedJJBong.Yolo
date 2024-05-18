using CanCanNeedJJBong.Yolo.Core.Basic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace CanCanNeedJJBong.Yolo.Core.TaskModelStrategy;

/// <summary>
/// 任务模式推理策略接口
/// </summary>
public interface ITaskModelInferenceStrategy
{
    List<YoloData> ExecuteTask(YoloConfig yolo,IReadOnlyCollection<NamedOnnxValue> container,float confidenceDegree, float iouThreshold, bool allIou);
}