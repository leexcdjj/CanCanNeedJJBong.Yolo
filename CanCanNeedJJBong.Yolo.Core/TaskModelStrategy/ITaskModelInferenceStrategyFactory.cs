namespace CanCanNeedJJBong.Yolo.Core.TaskModelStrategy;

/// <summary>
/// 任务模式推理策略工厂接口
/// </summary>
public interface ITaskModelInferenceStrategyFactory
{
    /// <summary>
    /// 创建任务模式推理策略
    /// </summary>
    /// <param name="taskMode">任务模式</param>
    /// <param name="yoloVersion">Yolo版本</param>
    /// <returns></returns>
    ITaskModelInferenceStrategy CreateTaskModelStrategy(int taskMode,int yoloVersion);
}