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
    public List<YoloData> ExecuteTask(YoloConfig yolo, IReadOnlyCollection<NamedOnnxValue> container,
        float confidenceDegree, float iouThreshold, bool allIou)
    {
        // 运行模型并获取结果
        Tensor<float> data = yolo.ModelSession.Run(container).First().AsTensor<float>();
        List<YoloData> result = new List<YoloData>();

        // 判断数据维度是否为中间尺寸
        bool isMidSize = data.Dimensions[1] < data.Dimensions[2];

        if (isMidSize)
        {
            // 使用并发集合处理并行任务
            ConcurrentBag<YoloData> conList = new ConcurrentBag<YoloData>();
            Parallel.For(0, data.Dimensions[2], i =>
            {
                float maxConfidence = 0f;
                int maxIndex = -1;
                // 遍历每个检测结果，找到最大置信度的索引
                for (int j = 0; j < data.Dimensions[1] - 4 - yolo.SemanticSegmentationWidth - yolo.ActionWidth; j++)
                {
                    if (data[0, j + 4, i] >= confidenceDegree)
                    {
                        if (maxConfidence < data[0, j + 4, i])
                        {
                            maxConfidence = data[0, j + 4, i];
                            maxIndex = j;
                        }
                    }
                }

                // 如果找到有效索引，则添加到结果列表
                if (maxIndex != -1)
                {
                    float[] tobeAddData = new float[6];
                    YoloData temp = new YoloData();
                    tobeAddData[0] = data[0, 0, i];
                    tobeAddData[1] = data[0, 1, i];
                    tobeAddData[2] = data[0, 2, i];
                    tobeAddData[3] = data[0, 3, i];
                    tobeAddData[4] = maxConfidence;
                    tobeAddData[5] = maxIndex;
                    temp.BasicData = tobeAddData;
                    conList.Add(temp);
                }
            });

            // 转换并发集合为列表
            result = conList.ToList();
        }
        else
        {
            // 对于非中间尺寸的数据处理
            int outputSize = data.Dimensions[2];
            float[] dataArray = data.ToArray();
            for (int i = 0; i < dataArray.Length; i += outputSize)
            {
                float maxConfidence = 0f;
                int maxIndex = -1;
                for (int j = 0; j < outputSize - 4 - yolo.SemanticSegmentationWidth - yolo.ActionWidth; j++)
                {
                    if (dataArray[i + 4 + j] > confidenceDegree)
                    {
                        if (maxConfidence < dataArray[i + 4 + j])
                        {
                            maxConfidence = dataArray[i + 4 + j];
                            maxIndex = j;
                        }
                    }
                }

                if (maxIndex != -1)
                {
                    float[] tobeAddData = new float[6];
                    YoloData temp = new YoloData();
                    tobeAddData[0] = dataArray[i];
                    tobeAddData[1] = dataArray[i + 1];
                    tobeAddData[2] = dataArray[i + 2];
                    tobeAddData[3] = dataArray[i + 3];
                    tobeAddData[4] = maxConfidence;
                    tobeAddData[5] = maxIndex;
                    temp.BasicData = tobeAddData;
                    result.Add(temp);
                }
            }
        }

        // 应用非极大值抑制（NMS）过滤
        result = YoloHelper.NMSFilter(result, iouThreshold, allIou);

        return result;
    }
}