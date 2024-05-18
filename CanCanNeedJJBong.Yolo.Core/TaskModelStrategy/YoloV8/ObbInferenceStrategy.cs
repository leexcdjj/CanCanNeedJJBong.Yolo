using System.Collections.Concurrent;
using CanCanNeedJJBong.Yolo.Core.Basic;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace CanCanNeedJJBong.Yolo.Core.TaskModelStrategy.YoloV8;

/// <summary>
/// Obb模式
/// </summary>
public class ObbInferenceStrategy : ITaskModelInferenceStrategy
{
    public List<YoloData> ExecuteTask(Tensor<float> data, float confidenceDegree, float iouThreshold, bool allIou,
        Yolo yolo)
    {
       // 判断维度是否为中等大小
        bool isMidSize = data.Dimensions[1] < data.Dimensions[2];
        if (isMidSize)
        {
            ConcurrentBag<YoloData> result = new ConcurrentBag<YoloData>();
            int outputSize = data.Dimensions[1];

            // 使用并行处理每个维度
            Parallel.For(0, data.Dimensions[2], i =>
            {
                float tempConfidenceDegree = 0f;
                int index = -1;

                // 遍历数据，找到满足置信度条件的最大值及其索引
                for (int j = 0; j < data.Dimensions[1] - 5; j++)
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
                    float[] tobeAddData = new float[7];
                    YoloData tempData = new YoloData
                    {
                        BasicData = new float[]
                        {
                            data[0, 0, i],
                            data[0, 1, i],
                            data[0, 2, i],
                            data[0, 3, i],
                            tempConfidenceDegree,
                            index,
                            data[0, outputSize - 1, i]
                        }
                    };
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

                for (int j = 0; j < outputSize - 5; j++)
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
                    float[] tobeAddData = new float[7];
                    YoloData tempData = new YoloData
                    {
                        BasicData = new float[]
                        {
                            dataArray[i],
                            dataArray[i + 1],
                            dataArray[i + 2],
                            dataArray[i + 3],
                            tempConfidenceDegree,
                            index,
                            dataArray[i + outputSize - 1]
                        }
                    };
                    result.Add(tempData);
                }
            }

            return result;
        }
    }
}