using System.Collections.Concurrent;
using CanCanNeedJJBong.Yolo.Core.Basic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace CanCanNeedJJBong.Yolo.Core.TaskModelStrategy.YoloV8;

/// <summary>
/// YoloV8 pose模式(动作)
/// </summary>
public class PoseInferenceStrategyV8 : ITaskModelInferenceStrategy
{
    public List<YoloData> ExecuteTask(YoloConfig yolo,IReadOnlyCollection<NamedOnnxValue> container, float confidenceDegree, float iouThreshold, bool allIou)
    {
        var data = yolo.ModelSession.Run(container).First().AsTensor<float>();
        
        // 判断维度是否为中等大小
        bool isMidSize = data.Dimensions[1] < data.Dimensions[2];
        if (isMidSize)
        {
            ConcurrentBag<YoloData> result = new ConcurrentBag<YoloData>();

            // 使用并行处理每个维度
            Parallel.For(0, data.Dimensions[2], i =>
            {
                float tempConfidenceDegree = 0f;
                int index = -1;

                // 遍历数据，找到满足置信度条件的最大值及其索引
                for (int j = 0; j < data.Dimensions[1] - 4 - yolo.SemanticSegmentationWidth - yolo.ActionWidth; j++)
                {
                    float currentConfidence = data[0, j + 4, i];
                    if (currentConfidence >= confidenceDegree && currentConfidence > tempConfidenceDegree)
                    {
                        tempConfidenceDegree = currentConfidence;
                        index = j;
                    }
                }

                // 如果找到了合适的数据，添加到结果集中
                if (index != -1)
                {
                    float[] tobeAddData = new float[6];
                    YoloData tempData = new YoloData
                    {
                        BasicData = new float[]
                        {
                            data[0, 0, i],
                            data[0, 1, i],
                            data[0, 2, i],
                            data[0, 3, i],
                            tempConfidenceDegree,
                            index
                        }
                    };

                    // 获取姿势数据
                    Pose[] p = new Pose[yolo.ActionWidth / 3];
                    for (int ii = 0; ii < yolo.ActionWidth; ii += 3)
                    {
                        p[ii / 3] = new Pose
                        {
                            X = data[0, 5 + ii, i],
                            Y = data[0, 6 + ii, i],
                            V = data[0, 7 + ii, i]
                        };
                    }

                    tempData.PointKeys = p;
                    result.Add(tempData);
                }
            });

            return result.ToList();
        }
        else
        {
            List<YoloData> result = new List<YoloData>();
            float[] dataArray = data.ToArray();
            int outputSize = data.Dimensions[2];

            // 遍历数据，找到满足置信度条件的最大值及其索引
            for (int i = 0; i < dataArray.Length; i += outputSize)
            {
                float tempConfidenceDegree = 0f;
                int index = -1;

                for (int j = 0; j < outputSize - 4 - yolo.ActionWidth; j++)
                {
                    float currentConfidence = dataArray[i + 4 + j];
                    if (currentConfidence >= confidenceDegree && currentConfidence > tempConfidenceDegree)
                    {
                        tempConfidenceDegree = currentConfidence;
                        index = j;
                    }
                }

                // 如果找到了合适的数据，添加到结果集中
                if (index != -1)
                {
                    float[] tobeAddData = new float[6];
                    YoloData tempData = new YoloData
                    {
                        BasicData = new float[]
                        {
                            dataArray[i],
                            dataArray[i + 1],
                            dataArray[i + 2],
                            dataArray[i + 3],
                            tempConfidenceDegree,
                            index
                        }
                    };

                    // 获取姿势数据
                    Pose[] p = new Pose[yolo.ActionWidth / 3];
                    for (int ii = 0; ii < yolo.ActionWidth; ii += 3)
                    {
                        p[ii / 3] = new Pose
                        {
                            X = dataArray[i + 5 + ii],
                            Y = dataArray[i + 6 + ii],
                            V = dataArray[i + 7 + ii]
                        };
                    }

                    tempData.PointKeys = p;
                    result.Add(tempData);
                }
            }
            
            // NMS过滤
            result = YoloHelper.NMSFilter(result, iouThreshold, allIou);

            return result;
        }
    }
}