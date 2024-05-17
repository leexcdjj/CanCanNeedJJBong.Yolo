using System.Collections.Concurrent;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace CanCanNeedJJBong.Yolo.Core;

public class Yolo
{
    /// <summary>
    /// 模型会话
    /// </summary>
    public InferenceSession ModelSession { get; set; }

    /// <summary>
    /// 张量宽度
    /// </summary>
    public int TensorWidth { get; set; }

    /// <summary>
    /// 张量高度
    /// </summary>
    public int TensorHeight { get; set; }

    /// <summary>
    /// 模型输入名
    /// </summary>
    public string ModelInputName { get; set; }

    /// <summary>
    /// 模型输出名
    /// </summary>summary
    public string ModelOutputName { get; set; }

    /// <summary>
    /// 输入张量信息
    /// </summary>
    public int[] InputTensorInfo { get; set; }

    /// <summary>
    /// 输出张量信息
    /// </summary>
    public int[] OutputTensorInfo { get; set; }

    /// <summary>
    /// 输出张量信息2_分割
    /// </summary>
    public int[] OutputTensorInfo2_Segmentation { get; set; }

    /// <summary>
    /// 推理图片宽度
    /// </summary>
    public int InferenceImageWidth { get; set; }

    /// <summary>
    /// 推理图片高度
    /// </summary>
    public int InferenceImageHeight { get; set; }

    /// <summary>
    /// 输入张量
    /// </summary>
    public DenseTensor<float> InputTensor { get; set; }

    /// <summary>
    /// YOLO版本
    /// </summary>
    public int YoloVersion { get; set; }

    /// <summary>
    /// mask缩放比例W
    /// </summary>
    public float MaskScaleRatioW { get; set; } = 0;

    /// <summary>
    /// mask缩放比例H
    /// </summary>
    public float MaskScaleRatioH { get; set; } = 0;

    /// <summary>
    /// 模型版本
    /// </summary>
    public string ModelVersion { get; set; } = "";

    /// <summary>
    /// 任务类型
    /// </summary>
    public string TaskType { get; set; } = "";

    /// <summary>
    /// 语义分割宽度
    /// </summary>
    public int SemanticSegmentationWidth { get; set; } = 0;

    /// <summary>
    /// 动作宽度
    /// </summary>
    public int ActionWidth { get; set; } = 0;

    /// <summary>
    /// 缩放比例
    /// </summary>
    public float ScaleRatio { get; set; } = 1;

    /// <summary>
    /// 标签组
    /// </summary>
    public string[] LabelGroup { get; set; }

    /// <summary>
    /// 执行任务模式
    /// </summary>
    private int ExecutionTaskMode = 0;

    /// <summary>
    /// 任务模式
    /// </summary>
    public int TaskMode
    {
        get { return ExecutionTaskMode; }
        set
        {
            if (TaskType == "classify")
            {
                ExecutionTaskMode = 0;
            }
            //检测
            else if (TaskType == "detect")
            {
                ExecutionTaskMode = 1;
            }
            //分割
            else if (TaskType == "segment")
            {
                if (value == 1 || value == 2 || value == 3)
                {
                    ExecutionTaskMode = value;
                }
                else
                {
                    ExecutionTaskMode = 3;
                }
            }
            //动作
            else if (TaskType == "pose")
            {
                if (value == 1 || value == 4 || value == 5)
                {
                    ExecutionTaskMode = value;
                }
                else
                {
                    ExecutionTaskMode = 5;
                }
            }
            else if (TaskType == "obb")
            {
                if (value == 6)
                {
                    ExecutionTaskMode = value;
                }
                else
                {
                    ExecutionTaskMode = 6;
                }
            }
            else
            {
                //如果什么都数据都没有,默认是检测模型
                ExecutionTaskMode = 1;
            }
        }
    }

    /// <summary>
    /// 推理主要函数
    /// </summary>
    /// <param name="imgData">图片数据</param>
    /// <param name="confidenceDegree">置信度:0-1的小数,数值越高越快,检测精度要求也会越高.</param>
    /// <param name="iouThreshold">iou阈值:0-1的小数,数值越高出现重复框的概率就越大,该值代表允许方框的最大重复面积</param>
    /// <param name="allIou">全局iou:false代表不同种类的框按允许重叠,true代表所有不同标签种类的框都要按照iou阈值不准重叠</param>  
    /// <param name="preprocessingMode">预处理模式:0表示高精度模式,对小物体有更高精度的检测;1表示高速模式,尤其是对大图像,大幅提高推理速度</param>   
    /// <returns>返回列表的基本数据格式{目标中心点x,目标中心点y,识别宽度,识别高度,置信度,标签索引}</returns>
    public List<YoloData> ModelReasoning(Bitmap imgData, float confidenceDegree = 0.5f, float iouThreshold = 0.3f, bool allIou = false, int preprocessingMode = 1)
    {
        InputTensor = new DenseTensor<float>(InputTensorInfo);
        var sp = InputTensor.Buffer.Span;
        ScaleRatio = 1;
        InferenceImageWidth = imgData.Width;
        InferenceImageHeight = imgData.Height;
        
        if (preprocessingMode == 0)
        {
            imgData = PictureZoom(imgData);
            InputTensor = PicWriTensor_Memory(imgData, InputTensorInfo);
        }
        if (preprocessingMode == 1)
        {
            InputTensor = NoInterpolationWriteTensor(imgData, InputTensorInfo);
        }
        
        //容器
        IReadOnlyCollection<NamedOnnxValue> container = new List<NamedOnnxValue>
            { NamedOnnxValue.CreateFromTensor(ModelInputName, InputTensor) };
        
        Tensor<float> output0;
        Tensor<float> output1;
        
        // 过滤数据数组
        List<YoloData> filterDataList;
        List<YoloData> result = new List<YoloData>();
        
        if (ExecutionTaskMode == 0)
        {
            output0 = ModelSession.Run(container).First().AsTensor<float>();
            result = ConfidenceFilter_Class(output0, confidenceDegree);
        }
        else if (ExecutionTaskMode == 1)
        {
            output0 = ModelSession.Run(container).First().AsTensor<float>();
            if (YoloVersion == 8)
            {
                filterDataList = ConfidenceFilter_YoloV89_Detection(output0, confidenceDegree);
            }
            else if (YoloVersion == 5)
            {
                filterDataList = new List<YoloData>();
                // todo：暂不支持yolov5
                // filterDataList = 置信度过滤_yolo5检测(output0, confidenceDegree);
            }
            else
            {
                filterDataList = new List<YoloData>();
                // todo：暂不支持yolov6
                // filterDataList = 置信度过滤_yolo6检测(output0, confidenceDegree);
            }

            result = NMSFilter(filterDataList, iouThreshold, allIou);
        }
        else if (ExecutionTaskMode == 2 || ExecutionTaskMode == 3)
        {
            // 返回数据
            var list = ModelSession.Run(container);
            output0 = list.First().AsTensor<float>();
            output1 = list.ElementAtOrDefault(1)?.AsTensor<float>();
            if (YoloVersion == 8)
            {
                filterDataList = ConfidenceFilter_YoloV8_Split(output0, confidenceDegree);
            }
            else
            {
                filterDataList = new List<YoloData>();
                // todo: 暂不支持yolov5分割
                // filterDataList = 置信度过滤_yolo5分割(output0, confidenceDegree);
            }

            result = NMSFilter(filterDataList, iouThreshold, allIou);
            ReductionMask(result, output1);
        }
        else if (ExecutionTaskMode == 4 || ExecutionTaskMode == 5)
        {
            output0 = ModelSession.Run(container).First().AsTensor<float>();
            filterDataList = ConfidenceFilter_Action(output0, confidenceDegree);
            result = NMSFilter(filterDataList, iouThreshold, allIou);
        }
        else if (ExecutionTaskMode == 6)
        {
            output0 = ModelSession.Run(container).First().AsTensor<float>();
            filterDataList = ConfidenceFilter_OBB(output0, confidenceDegree);
            result = NMSFilter(filterDataList, iouThreshold, allIou);
        }

        RestoreCoordinates(result);
        if (ExecutionTaskMode != 0)
        {
            RemoveCoordinates(result);
        }

        return result;
    }

    #region 图片缩放
    /// <summary>
    /// 图片缩放
    /// </summary>
    /// <param name="imgData">图片数据</param>
    /// <returns>缩放后的图片</returns>
    private Bitmap PictureZoom(Bitmap imgData)
    {
        // 缩放图片的目标宽度和高度
        float targetWidth = InferenceImageWidth;
        float targetHeight = InferenceImageHeight;

        // 计算缩放比例
        float scaleRatio = CalculateScaleRatio(targetWidth, targetHeight);

        // 按比例调整目标宽度和高度
        targetWidth *= scaleRatio;
        targetHeight *= scaleRatio;

        // 创建缩放后的图片
        Bitmap zoomImg = new Bitmap((int)targetWidth, (int)targetHeight);

        // 绘制缩放后的图片
        using (Graphics graphics = Graphics.FromImage(zoomImg))
        {
            // 设置插值模式为高质量双三次插值
            graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
            graphics.DrawImage(imgData, 0, 0, targetWidth, targetHeight);
        }

        return zoomImg;
    }

    /// <summary>
    /// 计算缩放比例
    /// </summary>
    /// <param name="targetWidth">目标宽度</param>
    /// <param name="targetHeight">目标高度</param>
    /// <returns>缩放比例</returns>
    private float CalculateScaleRatio(float targetWidth, float targetHeight)
    {
        if (targetWidth > TensorWidth || targetHeight > TensorHeight)
        {
            // 选择较小的缩放比例
            return Math.Min(TensorWidth / targetWidth, TensorHeight / targetHeight);
        }
        return 1f;
    }
    
    #endregion
    
    
    /// <summary>
    /// 图片写到张量_内存并行
    /// </summary>
    /// <param name="imgData">图片数据</param>
    /// <param name="inputTensorInfo">输入张量信息</param>
    /// <returns></returns>
    private DenseTensor<float> PicWriTensor_Memory(Bitmap imgData, int[] inputTensorInfo)
    {
        int height = imgData.Height;
        int width = imgData.Width;
        
        // 内存图片数据
        BitmapData memoryImgData = imgData.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly,
            PixelFormat.Format24bppRgb);
        
        int stride = memoryImgData.Stride;
        IntPtr scan0 = memoryImgData.Scan0;
        
        // 临时数据
        float[,,] tempData = new float[inputTensorInfo[1], inputTensorInfo[2], inputTensorInfo[3]];
        
        try
        {
            Parallel.For(0, height, y =>
            {
                for (int x = 0; x < width; x++)
                {
                    IntPtr pixel = IntPtr.Add(scan0, y * stride + x * 3);
                    tempData[2, y, x] = Marshal.ReadByte(pixel) / 255f;
                    pixel = IntPtr.Add(pixel, 1);
                    tempData[1, y, x] = Marshal.ReadByte(pixel) / 255f;
                    pixel = IntPtr.Add(pixel, 1);
                    tempData[0, y, x] = Marshal.ReadByte(pixel) / 255f;
                }
            });
        }
        finally
        {
            imgData.UnlockBits(memoryImgData);
        }
        
        // 展开临时数据
        float[] deploymentTempData = new float[inputTensorInfo[1] * inputTensorInfo[2] * inputTensorInfo[3]];
        Buffer.BlockCopy(tempData, 0, deploymentTempData, 0, deploymentTempData.Length * 4);
        
        return new DenseTensor<float>(deploymentTempData, inputTensorInfo);
    }
    
    /// <summary>
    /// 无插值写入张量
    /// </summary>
    /// <param name="imgData">图片数据</param>
    /// <param name="inputTensorInfo">输入张量信息</param>
    /// <returns></returns>
    private DenseTensor<float> NoInterpolationWriteTensor(Bitmap imgData, int[] inputTensorInfo)
    {
        // 内存图片数据
        BitmapData memoryPicData = imgData.LockBits(new Rectangle(0, 0, imgData.Width, imgData.Height), ImageLockMode.ReadOnly,
            PixelFormat.Format24bppRgb);
        
        int stride = memoryPicData.Stride;
        IntPtr scan0 = memoryPicData.Scan0;
        
        // 临时数据
        float[,,] tempData = new float[inputTensorInfo[1], inputTensorInfo[2], inputTensorInfo[3]];
        
        float zoomImgWidth = InferenceImageWidth;
        float zoomImgHeight = InferenceImageHeight;
        
        if (zoomImgWidth > TensorWidth || zoomImgHeight > TensorHeight)
        {
            ScaleRatio = (TensorWidth / zoomImgWidth) < (TensorHeight / zoomImgHeight) ? (TensorWidth / zoomImgWidth) : (TensorHeight / zoomImgHeight);
            zoomImgWidth = zoomImgWidth * ScaleRatio;
            zoomImgHeight = zoomImgHeight * ScaleRatio;
        }
        
        // x,y坐标
        int xCoordinate, yCoordinate;
        
        // 系数
        float coefficient = 1 / ScaleRatio;
        
        for (int y = 0; y < (int)zoomImgHeight; y++)
        {
            for (int x = 0; x < (int)zoomImgWidth; x++)
            {
                xCoordinate = (int)(x * coefficient);
                yCoordinate = (int)(y * coefficient);
                IntPtr pixel = IntPtr.Add(scan0, yCoordinate * stride + xCoordinate * 3);
                tempData[2, y, x] = Marshal.ReadByte(pixel) / 255f;
                pixel = IntPtr.Add(pixel, 1);
                tempData[1, y, x] = Marshal.ReadByte(pixel) / 255f;
                pixel = IntPtr.Add(pixel, 1);
                tempData[0, y, x] = Marshal.ReadByte(pixel) / 255f;
            }
        }

        imgData.UnlockBits(memoryPicData);
        float[] deploymentTempData = new float[inputTensorInfo[1] * inputTensorInfo[2] * inputTensorInfo[3]];
        Buffer.BlockCopy(tempData, 0, deploymentTempData, 0, deploymentTempData.Length * 4);
        return new DenseTensor<float>(deploymentTempData, inputTensorInfo);
    }
    
    /// <summary>
    /// 置信度过滤_分类
    /// </summary>
    /// <param name="data">数据</param>
    /// <param name="confidenceDegree">置信度</param>
    /// <returns></returns>
    private List<YoloData> ConfidenceFilter_Class(Tensor<float> data, float confidenceDegree)
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
        RankConfidenceDegree(result);
        return result;
    }
    
    /// <summary>
    /// 置信度排序
    /// </summary>
    /// <param name="data"></param>
    private void RankConfidenceDegree(List<YoloData> data)
    {
        if (data == null || data.Count == 0)
        {
            return;
        }

        if (data[0].BasicData.Length == 2)
        {
            // 使用内置的排序算法，根据 BasicData[0] 降序排序
            data.Sort((x, y) => y.BasicData[0].CompareTo(x.BasicData[0]));
        }
        else
        {
            // 使用内置的排序算法，根据 BasicData[4] 降序排序
            data.Sort((x, y) => y.BasicData[4].CompareTo(x.BasicData[4]));
        }
    }

    #region 计算交并比 CalcCrossMergeRatio

    /// <summary>
    /// 计算交并比（IoU）
    /// </summary>
    /// <param name="rectangle1">矩形1</param>
    /// <param name="rectangle2">矩形2</param>
    /// <returns>交并比（IoU）</returns>
    private float CalcCrossMergeRatio(float[] rectangle1, float[] rectangle2)
    {
        // 计算矩形1的边界
        float[] rect1Bounds = CalculateBounds(rectangle1);
        // 计算矩形2的边界
        float[] rect2Bounds = CalculateBounds(rectangle2);

        // 计算交集区域
        float intersectionArea = CalculateIntersectionArea(rect1Bounds, rect2Bounds);

        // 计算矩形1和矩形2的面积
        float area1 = CalculateArea(rect1Bounds);
        float area2 = CalculateArea(rect2Bounds);

        // 计算并集区域
        float unionArea = area1 + area2 - intersectionArea;

        // 返回交并比（IoU）
        return intersectionArea / unionArea;
    }

    /// <summary>
    /// 计算矩形的边界
    /// </summary>
    /// <param name="rectangle">矩形</param>
    /// <returns>矩形的边界数组 [left, top, right, bottom]</returns>
    private float[] CalculateBounds(float[] rectangle)
    {
        return new float[]
        {
            rectangle[0] - rectangle[2] / 2, // left
            rectangle[1] - rectangle[3] / 2, // top
            rectangle[0] + rectangle[2] / 2, // right
            rectangle[1] + rectangle[3] / 2 // bottom
        };
    }

    /// <summary>
    /// 计算两个矩形的交集面积
    /// </summary>
    /// <param name="rect1">矩形1的边界</param>
    /// <param name="rect2">矩形2的边界</param>
    /// <returns>交集面积</returns>
    private float CalculateIntersectionArea(float[] rect1, float[] rect2)
    {
        float leftBoundary = Math.Max(rect1[0], rect2[0]);
        float topBoundary = Math.Max(rect1[1], rect2[1]);
        float rightBoundary = Math.Min(rect1[2], rect2[2]);
        float bottomBoundary = Math.Min(rect1[3], rect2[3]);

        if (leftBoundary < rightBoundary && topBoundary < bottomBoundary)
        {
            return (rightBoundary - leftBoundary) * (bottomBoundary - topBoundary);
        }

        return 0;
    }

    /// <summary>
    /// 计算矩形的面积
    /// </summary>
    /// <param name="bounds">矩形的边界</param>
    /// <returns>矩形的面积</returns>
    private float CalculateArea(float[] bounds)
    {
        return (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]);
    }

    #endregion
    

    /// <summary>
    /// 置信度过滤_yolo8_9检测
    /// </summary>
    /// <param name="data">数据</param>
    /// <param name="confidenceDegree">置信度</param>
    /// <returns>过滤后的Yolo数据列表</returns>
    private List<YoloData> ConfidenceFilter_YoloV89_Detection(Tensor<float> data, float confidenceDegree)
    {
        // 是否中间是尺寸
        bool midIsSize = data.Dimensions[1] < data.Dimensions[2];
        int outputSize = midIsSize ? data.Dimensions[2] : data.Dimensions[1];
        int numDetections = midIsSize ? data.Dimensions[2] : data.Dimensions[0];
        int confIndex = 4 + SemanticSegmentationWidth + ActionWidth;

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

            return result;
        }
    }

    /// <summary>
    /// nms过滤
    /// </summary>
    /// <param name="filterDataList">首次过滤数组</param>
    /// <param name="iouThreshold">iou阈值</param>
    /// <param name="allIou">全局iou</param>
    /// <returns>经过nms过滤后的Yolo数据列表</returns>
    private List<YoloData> NMSFilter(List<YoloData> filterDataList, float iouThreshold, bool allIou)
    {
        // 对数据按置信度降序排序
        RankConfidenceDegree(filterDataList);

        // 保存过滤后的结果
        List<YoloData> result = new List<YoloData>();

        foreach (var data in filterDataList)
        {
            bool isFilter = true;

            foreach (var res in result)
            {
                // 判断是否计算全局iou或属于同一类
                if (allIou || data.BasicData[5] == res.BasicData[5])
                {
                    // 计算两个矩形的交并比
                    float iou = CalcCrossMergeRatio(data.BasicData, res.BasicData);

                    // 如果iou超过阈值，设置isFilter为false
                    if (iou > iouThreshold)
                    {
                        isFilter = false;
                        break;
                    }
                }
            }

            // 如果通过过滤，将数据添加到结果中
            if (isFilter) result.Add(data);
        }

        return result;
    }

    /// <summary>
    /// 置信度过滤_yolo8分割
    /// </summary>
    /// <param name="data">数据</param>
    /// <param name="confidenceDegree">置信度</param>
    /// <returns>经过置信度过滤后的Yolo数据列表</returns>
    private List<YoloData> ConfidenceFilter_YoloV8_Split(Tensor<float> data, float confidenceDegree)
    {
        // 判断中间是否是尺寸
        bool isMidSize = data.Dimensions[1] < data.Dimensions[2];
        int dim1 = data.Dimensions[1];
        int dim2 = data.Dimensions[2];
        int semanticOffset = dim1 - SemanticSegmentationWidth;

        if (isMidSize)
        {
            ConcurrentBag<YoloData> result = new ConcurrentBag<YoloData>();

            // 并行处理每个检测结果
            Parallel.For(0, dim2, i =>
            {
                float maxConfidence = 0f;
                int maxIndex = -1;

                // 遍历置信度，找出最高的置信度和对应的索引
                for (int j = 0; j < dim1 - 4 - SemanticSegmentationWidth; j++)
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
                    for (int ii = 0; ii < SemanticSegmentationWidth; ii++)
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
                for (int j = 0; j < outputSize - 4 - SemanticSegmentationWidth; j++)
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
                    for (int ii = 0; ii < SemanticSegmentationWidth; ii++)
                    {
                        int pos = i + outputSize - SemanticSegmentationWidth + ii;
                        mask.At<float>(0, ii) = dataArray[pos];
                    }

                    temp.MaskData = mask;
                    temp.BasicData = tobeAddData;
                    result.Add(temp);
                }
            }

            return result;
        }
    }

    /// <summary>
    /// 还原掩膜
    /// </summary>
    /// <param name="data">数据</param>
    /// <param name="output1">输出张量</param>
    private void ReductionMask(List<YoloData> data, Tensor<float>? output1)
    {
        // 将输出张量转换为矩阵
        Mat ot1 = new Mat(SemanticSegmentationWidth,
            OutputTensorInfo2_Segmentation[2] * OutputTensorInfo2_Segmentation[3], MatType.CV_32F, output1.ToArray());

        foreach (var yoloData in data)
        {
            // 原始mask
            Mat initMask = yoloData.MaskData * ot1;

            // 对mask进行并行处理，应用Sigmoid函数
            Parallel.For(0, initMask.Cols,
                col => { initMask.At<float>(0, col) = Sigmoid(initMask.At<float>(0, col)); });

            // 重塑mask
            Mat newMask = initMask.Reshape(1, OutputTensorInfo2_Segmentation[2], OutputTensorInfo2_Segmentation[3]);

            // 计算掩膜的左上角和宽高
            int maskX1 = Math.Max(0, (int)((yoloData.BasicData[0] - yoloData.BasicData[2] / 2) * MaskScaleRatioW));
            int maskY1 = Math.Max(0, (int)((yoloData.BasicData[1] - yoloData.BasicData[3] / 2) * MaskScaleRatioH));
            int maskWidth = (int)(yoloData.BasicData[2] * MaskScaleRatioW);
            int maskHeight = (int)(yoloData.BasicData[3] * MaskScaleRatioH);

            // 限制宽高在输出张量的范围内
            if (maskX1 + maskWidth > OutputTensorInfo2_Segmentation[3])
                maskWidth = OutputTensorInfo2_Segmentation[3] - maskX1;
            if (maskY1 + maskHeight > OutputTensorInfo2_Segmentation[2])
                maskHeight = OutputTensorInfo2_Segmentation[2] - maskY1;

            // 定义裁剪区域
            Rect rect = new Rect(maskX1, maskY1, maskWidth, maskHeight);

            // 裁剪后的掩膜
            Mat afterMat = new Mat(newMask, rect);

            // 还原原图掩膜
            Mat restoredMask = new Mat();

            // 计算放大后的宽高
            int enlargedWidth = (int)(afterMat.Width / MaskScaleRatioW / ScaleRatio);
            int enlargedHeight = (int)(afterMat.Height / MaskScaleRatioH / ScaleRatio);

            // 调整掩膜大小
            Cv2.Resize(afterMat, restoredMask, new OpenCvSharp.Size(enlargedWidth, enlargedHeight));
            // 阈值处理
            Cv2.Threshold(restoredMask, restoredMask, 0.5, 1, ThresholdTypes.Binary);

            // 将还原后的掩膜赋值回数据
            yoloData.MaskData = restoredMask;
        }
    }
    
    private float Sigmoid(float value)
    {
        return 1 / (1 + (float)Math.Exp(-value));
    }

    /// <summary>
    /// 置信度过滤_动作
    /// </summary>
    /// <param name="data">数据</param>
    /// <param name="confidenceDegree">置信度</param>
    /// <returns></returns>
    private List<YoloData> ConfidenceFilter_Action(Tensor<float> data, float confidenceDegree)
    {
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
                for (int j = 0; j < data.Dimensions[1] - 4 - SemanticSegmentationWidth - ActionWidth; j++)
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
                    Pose[] p = new Pose[ActionWidth / 3];
                    for (int ii = 0; ii < ActionWidth; ii += 3)
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

                for (int j = 0; j < outputSize - 4 - ActionWidth; j++)
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
                    Pose[] p = new Pose[ActionWidth / 3];
                    for (int ii = 0; ii < ActionWidth; ii += 3)
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

            return result;
        }
    }

    /// <summary>
    /// 置信度过滤_obb
    /// </summary>
    /// <param name="data">数据</param>
    /// <param name="confidenceDegree">置信度</param>
    /// <returns></returns>
    private List<YoloData> ConfidenceFilter_OBB(Tensor<float> data, float confidenceDegree)
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
    
    /// <summary>
    /// 还原返回坐标
    /// </summary>
    /// <param name="data">数据列表</param>
    private void RestoreCoordinates(List<YoloData> data)
    {
        if (data.Count == 0) return;

        // 如果 BasicData 长度大于 2，调整坐标比例
        if (data[0].BasicData.Length > 2)
        {
            foreach (var item in data)
            {
                for (int j = 0; j < 4; j++)
                {
                    item.BasicData[j] /= ScaleRatio;
                }
            }
        }

        // 如果 PointKeys 不为空，调整坐标比例
        if (data[0].PointKeys != null)
        {
            foreach (var item in data)
            {
                foreach (var point in item.PointKeys)
                {
                    point.X /= ScaleRatio;
                    point.Y /= ScaleRatio;
                }
            }
        }
    }

    /// <summary>
    /// 去除越界坐标
    /// </summary>
    /// <param name="data">数据列表</param>
    private void RemoveCoordinates(List<YoloData> data)
    {
        // 倒序移除越界数据
        for (int i = data.Count - 1; i >= 0; i--)
        {
            // 如果数据的任何一个坐标超出张量范围，移除该数据
            if (data[i].BasicData[0] > TensorWidth ||
                data[i].BasicData[1] > TensorHeight ||
                data[i].BasicData[2] > TensorWidth ||
                data[i].BasicData[3] > TensorHeight)
            {
                data.RemoveAt(i);
            }
        }
    }
}