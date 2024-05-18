using System.Collections.Concurrent;
using CanCanNeedJJBong.Yolo.Core.Basic;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace CanCanNeedJJBong.Yolo.Core.TaskModelStrategy.YoloV8;

/// <summary>
/// YoloV8 segment模式(分割)
/// </summary>
public class SegmentInferenceStrategyV8 : ITaskModelInferenceStrategy
{
    public List<YoloData> ExecuteTask(Tensor<float> data, float confidenceDegree, float iouThreshold, bool allIou,
        Yolo yolo)
    {
        // 判断中间是否是尺寸
        bool isMidSize = data.Dimensions[1] < data.Dimensions[2];
        int dim1 = data.Dimensions[1];
        int dim2 = data.Dimensions[2];
        int semanticOffset = dim1 - yolo.SemanticSegmentationWidth;

        if (isMidSize)
        {
            ConcurrentBag<YoloData> result = new ConcurrentBag<YoloData>();

            // 并行处理每个检测结果
            Parallel.For(0, dim2, i =>
            {
                float maxConfidence = 0f;
                int maxIndex = -1;

                // 遍历置信度，找出最高的置信度和对应的索引
                for (int j = 0; j < dim1 - 4 - yolo.SemanticSegmentationWidth; j++)
                {
                    float confidence = data[0, j + 4, i];
                    if (confidence >= confidenceDegree && confidence > maxConfidence)
                    {
                        maxConfidence = confidence;
                        maxIndex = j;
                    }
                }

                if (maxIndex != -1)
                {
                    // 准备要添加的数据
                    float[] tobeAddData = new float[6];
                    YoloData temp = new YoloData();
                    Mat mask = new Mat(1, 32, MatType.CV_32F);

                    tobeAddData[0] = data[0, 0, i];
                    tobeAddData[1] = data[0, 1, i];
                    tobeAddData[2] = data[0, 2, i];
                    tobeAddData[3] = data[0, 3, i];
                    tobeAddData[4] = maxConfidence;
                    tobeAddData[5] = maxIndex;

                    // 填充mask数据
                    for (int ii = 0; ii < yolo.SemanticSegmentationWidth; ii++)
                    {
                        int pos = semanticOffset + ii;
                        mask.At<float>(0, ii) = data[0, pos, i];
                    }

                    temp.MaskData = mask;
                    temp.BasicData = tobeAddData;
                    result.Add(temp);
                }
            });

            return result.ToList();
        }
        else
        {
            List<YoloData> result = new List<YoloData>();
            int outputSize = dim2;
            float[] dataArray = data.ToArray();

            for (int i = 0; i < dataArray.Length; i += outputSize)
            {
                float maxConfidence = 0f;
                int maxIndex = -1;

                // 遍历置信度，找出最高的置信度和对应的索引
                for (int j = 0; j < outputSize - 4 - yolo.SemanticSegmentationWidth; j++)
                {
                    float confidence = dataArray[i + 4 + j];
                    if (confidence > confidenceDegree && confidence > maxConfidence)
                    {
                        maxConfidence = confidence;
                        maxIndex = j;
                    }
                }

                if (maxIndex != -1)
                {
                    // 准备要添加的数据
                    float[] tobeAddData = new float[6];
                    YoloData temp = new YoloData();
                    Mat mask = new Mat(1, 32, MatType.CV_32F);

                    tobeAddData[0] = dataArray[i];
                    tobeAddData[1] = dataArray[i + 1];
                    tobeAddData[2] = dataArray[i + 2];
                    tobeAddData[3] = dataArray[i + 3];
                    tobeAddData[4] = maxConfidence;
                    tobeAddData[5] = maxIndex;

                    // 填充mask数据
                    for (int ii = 0; ii < yolo.SemanticSegmentationWidth; ii++)
                    {
                        int pos = i + outputSize - yolo.SemanticSegmentationWidth + ii;
                        mask.At<float>(0, ii) = dataArray[pos];
                    }

                    temp.MaskData = mask;
                    temp.BasicData = tobeAddData;
                    result.Add(temp);
                }
            }
            
            // NMS过滤
            result = YoloHelper.NMSFilter(result, iouThreshold, allIou);

            return result;
        }
    }
}