using CanCanNeedJJBong.Yolo.Core.Basic;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace CanCanNeedJJBong.Yolo.Core.TaskModelStrategy.YoloV8;

/// <summary>
/// YoloV8 classify模式
/// </summary>
public class ClassifyInferenceStrategyV8 : ITaskModelInferenceStrategy
{
    public List<YoloData> ExecuteTask(Tensor<float> data, float confidenceDegree, float iouThreshold, bool allIou, Yolo yolo)
    {
        List<YoloData> result = new List<YoloData>();

        for (int i = 0; i < data.Dimensions[1]; i++)
        {
            if (data[0, i] >= confidenceDegree)
            {
                //过滤信息  
                float[] filterMessage = new float[2];

                YoloData temp = new YoloData();

                //标签的置信度
                filterMessage[0] = data[0, i];
                //标签的索引
                filterMessage[1] = i;
                temp.BasicData = filterMessage;
                result.Add(temp);
            }
        }

        // 置信度排序
        YoloHelper.RankConfidenceDegree(result);
        return result;
    }
}