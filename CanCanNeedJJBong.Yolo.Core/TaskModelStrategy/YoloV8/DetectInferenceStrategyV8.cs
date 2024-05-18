using System.Collections.Concurrent;
using CanCanNeedJJBong.Yolo.Core.Basic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace CanCanNeedJJBong.Yolo.Core.TaskModelStrategy.YoloV8;

/// <summary>
/// YoloV8 detect模式(检测)
/// </summary>
public class DetectInferenceStrategyV8 : ITaskModelInferenceStrategy
{
    public List<YoloData> ExecuteTask(YoloConfig yolo,IReadOnlyCollection<NamedOnnxValue> container, float confidenceDegree, float iouThreshold, bool allIou)
    {
        var data = yolo.ModelSession.Run(container).First().AsTensor<float>();
        
        // 是否中间是尺寸
        bool midIsSize = data.Dimensions[1] < data.Dimensions[2];
        int outputSize = midIsSize ? data.Dimensions[2] : data.Dimensions[1];
        int numDetections = midIsSize ? data.Dimensions[2] : data.Dimensions[0];
        int confIndex = 4 + yolo.SemanticSegmentationWidth + yolo.ActionWidth;

        if (midIsSize)
        {
            var result = new ConcurrentBag<YoloData>();

            // 并行处理每个检测结果
            Parallel.For(0, numDetections, i =>
            {
                float tempConfidenceDegree = 0f;
                int index = -1;

                // 遍历置信度，找出最高的置信度和对应的索引
                for (int j = 0; j < outputSize - confIndex; j++)
                {
                    if (data[0, j + 4, i] >= confidenceDegree)
                    {
                        if (tempConfidenceDegree < data[0, j + 4, i])
                        {
                            tempConfidenceDegree = data[0, j + 4, i];
                            index = j;
                        }
                    }
                }

                if (index != -1)
                {
                    // 将结果添加到并发集合中
                    var temp = new YoloData
                    {
                        BasicData = new float[6]
                        {
                            data[0, 0, i],
                            data[0, 1, i],
                            data[0, 2, i],
                            data[0, 3, i],
                            tempConfidenceDegree,
                            index
                        }
                    };
                    result.Add(temp);
                }
            });

            return result.ToList();
        }
        else
        {
            var result = new List<YoloData>();
            var dataArray = data.ToArray();

            // 处理每个检测结果
            for (int i = 0; i < dataArray.Length; i += outputSize)
            {
                float tempConfidenceDegree = 0f;
                int index = -1;

                // 遍历置信度，找出最高的置信度和对应的索引
                for (int j = 0; j < outputSize - confIndex; j++)
                {
                    if (dataArray[i + 4 + j] > confidenceDegree)
                    {
                        if (tempConfidenceDegree < dataArray[i + 4 + j])
                        {
                            tempConfidenceDegree = dataArray[i + 4 + j];
                            index = j;
                        }
                    }
                }

                if (index != -1)
                {
                    // 将结果添加到列表中
                    var temp = new YoloData
                    {
                        BasicData = new float[6]
                        {
                            dataArray[i],
                            dataArray[i + 1],
                            dataArray[i + 2],
                            dataArray[i + 3],
                            tempConfidenceDegree,
                            index
                        }
                    };
                    result.Add(temp);
                }
            }
            
            // NMS过滤
            result = YoloHelper.NMSFilter(result, iouThreshold, allIou);

            return result;
        }
    }
}