using CanCanNeedJJBong.Yolo.Core.TaskModelStrategy.YoloV5;
using CanCanNeedJJBong.Yolo.Core.TaskModelStrategy.YoloV6;
using CanCanNeedJJBong.Yolo.Core.TaskModelStrategy.YoloV8;

namespace CanCanNeedJJBong.Yolo.Core.TaskModelStrategy;

/// <summary>
/// 任务模式推理策略工厂
/// </summary>
public class TaskModelInferenceStrategyFactory : ITaskModelInferenceStrategyFactory
{
    /// <summary>
    /// 创建任务模式推理策略
    /// </summary>
    /// <param name="taskMode">任务模式</param>
    /// <param name="yoloVersion">Yolo版本</param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    public ITaskModelInferenceStrategy CreateTaskModelStrategy(int taskMode, int yoloVersion)
    {
        ITaskModelInferenceStrategy strategy = null;
        
        switch (taskMode)
        {
            case 0:
                strategy = new ClassifyInferenceStrategyV8();
                break;
            case 1:
                if (yoloVersion == 8)
                {
                    strategy = new DetectInferenceStrategyV8();
                }

                if (yoloVersion == 5)
                {
                    strategy = new DetectInferenceStrategyV5();
                }
                
                if (yoloVersion == 6)
                {
                    strategy = new DetectInferenceStrategyV6();
                }
                
                break;
            case 2:
            case 3:
                if (yoloVersion == 8)
                {
                    strategy = new SegmentInferenceStrategyV8();
                }
                else
                {
                    strategy = new SegmentInferenceStrategyV5();
                }
                
                break;
            case 4:
            case 5:
                strategy = new PoseInferenceStrategyV8();
                break;
            case 6:
                strategy = new ObbInferenceStrategyV8();
                break;
            default:
                throw new Exception($"未实现任务模式：{taskMode}，yolo版本：{yoloVersion}的策略类");
        }

        return strategy;
    }
}