using System.Drawing;
using CanCanNeedJJBong.Yolo.Core.Basic;
using CanCanNeedJJBong.Yolo.Core.Builder;
using CanCanNeedJJBong.Yolo.Core.TaskModelStrategy;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace CanCanNeedJJBong.Yolo.Core;

public class YoloService
{
    private YoloConfig Config { get; set; }
    private ITaskModelInferenceStrategyFactory TaskModelInferenceStrategyFactory { get; set; }

    public YoloService(string modelPath, bool enableGpu, int gpuIndex, int yoloVersion)
    {
        enableGpu = false;
        gpuIndex = 0;
        YoloConfigBuilder builder = new YoloConfigBuilder();
        Config = builder.SetModelSession(modelPath, enableGpu, gpuIndex)
            .SetModelMetadata()
            .SetInferenceParameters(yoloVersion)
            .GetYoloConfig();

        TaskModelInferenceStrategyFactory = new TaskModelInferenceStrategyFactory();
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
    public List<YoloData> ModelReasoning(Bitmap imgData, float confidenceDegree = 0.5f, float iouThreshold = 0.3f,
        bool allIou = false, int preprocessingMode = 1)
    {
        Config.InputTensor = new DenseTensor<float>(Config.InputTensorInfo);
        var sp = Config.InputTensor.Buffer.Span;
        Config.ScaleRatio = 1;
        Config.InferenceImageWidth = imgData.Width;
        Config.InferenceImageHeight = imgData.Height;

        if (preprocessingMode == 0)
        {
            imgData = YoloHelper.PictureZoom(imgData, Config);
            Config.InputTensor = YoloHelper.PicWriTensor_Memory(imgData, Config.InputTensorInfo);
        }

        if (preprocessingMode == 1)
        {
            Config.InputTensor = YoloHelper.NoInterpolationWriteTensor(imgData, Config);
        }

        //容器
        IReadOnlyCollection<NamedOnnxValue> container = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(Config.ModelInputName, Config.InputTensor)
        };

        // 过滤数据数组
        List<YoloData> result = new List<YoloData>();

        // 获取任务模式处理策略
        var taskModelStrategy =
            TaskModelInferenceStrategyFactory.CreateTaskModelStrategy(Config.ExecutionTaskMode, Config.YoloVersion);

        // 策略处理
        result = taskModelStrategy.ExecuteTask(Config, container, confidenceDegree, iouThreshold, allIou);

        YoloHelper.RestoreCoordinates(result, Config);

        if (Config.ExecutionTaskMode != 0)
        {
            YoloHelper.RemoveCoordinates(result, Config);
        }

        return result;
    }

    /// <summary>
    /// 在原图像上绘制推理结果,同时受任务模式影响，如在分割模型上指定任务模式为检测，那么调用该方法生成的图像也只会画框，不会出现掩膜
    /// </summary>
    /// <param name="imgData">图片:原图像</param>
    /// <param name="data">yolo返回数据；由处理坐标返回的坐标组</param>
    /// <param name="lableGroup">标签组：类属性 标签组 或者手动传入一个数组</param>
    /// <param name="BorderBrush">边框画笔：指定一个边框画笔,如果为空则根据图片尺寸自适应</param>
    /// <param name="font">字体：指定一个字体,如果为空则根据图片尺寸自适应</param>
    /// <param name="textColorBrush">文字颜色笔刷：指定一个文字颜色笔刷,默认为黑</param>
    /// <param name="textBackgroundBrush">文字底色笔刷：指定一个文字底色笔刷,默认为橙色</param>
    /// <param name="segmentationMaskRandomColor">分割掩膜随机色：每一个目标都使用随机色掩膜,默认为真,为假时,统一使用绿色。当下个参数指定了掩膜颜色时,该参数失效</param>
    /// <param name="specifyLabelMaskColor">指定标签掩膜颜色：使用Color的ARGB颜色数组为同一个标签类别指定相同的颜色，当提供的数组的数量小于的标签的实际索引时，会使用默认的绿色，所以请确保数组的颜色数量大于标签数量</param>
    /// <param name="noMaskBcolor">非掩膜背景色：非掩膜的背景蒙版填充,默认无。与掩膜色相同,使用ARGB颜色,注意,该颜色的透明度会与掩膜颜色的透明度叠加,需自行调整到一个合理的值</param>
    /// <param name="classShowCount">分类显示数量：分类模型显示分类标签的数量,默认为5,即最多显示5个,分类标签的文字颜色、大小、底色,可直接从前面的参数指定</param>
    /// <param name="credibilityThresholdKeyPoints">关键点可信度阈值：用于决定显示关键点的可信度阈值,通常默认0.5,即高于0.5就显示,低于0.5就不显示,通常预测的点在框外的话就会低于0.5</param>
    /// <returns>返回绘制后的图像</returns>
    public Image GenerateIma(Image imgData, List<YoloData> data, string[] lableGroup, Pen? BorderBrush = null,
        Font font = null,
        SolidBrush textColorBrush = null, SolidBrush textBackgroundBrush = null,
        bool segmentationMaskRandomColor = true, Color[] specifyLabelMaskColor = null,
        Color? noMaskBcolor = null, int classShowCount = 5, float credibilityThresholdKeyPoints = 0.5f)
    {
        Bitmap reult = new Bitmap(imgData.Width, imgData.Height);

        Graphics gra = Graphics.FromImage(reult);
        gra.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAlias;
        if (BorderBrush == null)
        {
            // 画笔宽
            int brushWidth = (imgData.Width > imgData.Height ? imgData.Height : imgData.Width) / 135;

            BorderBrush = new Pen(Color.BlueViolet, brushWidth);
        }

        if (font == null)
        {
            // 字体宽
            int fontWidth = (imgData.Width > imgData.Height ? imgData.Height : imgData.Width) / 38;

            font = new Font("宋体", fontWidth, FontStyle.Bold);
        }

        if (textColorBrush == null) textColorBrush = new SolidBrush(Color.Black);
        if (textBackgroundBrush == null) textBackgroundBrush = new SolidBrush(Color.Orange);
        float textWidth;
        float textHeight;
        gra.DrawImage(imgData, 0, 0, imgData.Width, imgData.Height);
        string writeValue;

        //分类
        if (Config.ExecutionTaskMode == 0)
        {
            YoloHelper.RestoreDrawCoordinates(data);

            float x = 10;
            float y = 10;
            for (int i = 0; i < data.Count; i++)
            {
                if (i >= classShowCount) break;
                int lableIndex = (int)data[i].BasicData[1];
                string confidenceDegree = data[i].BasicData[0].ToString("_0.00");
                string lableName;
                if (lableIndex + 1 > lableGroup.Length)
                {
                    lableName = "无类别名称";
                }
                else
                {
                    lableName = lableGroup[lableIndex];
                }

                writeValue = lableName + confidenceDegree;
                textWidth = gra.MeasureString(writeValue + "_0.00", font).Width;
                textHeight = gra.MeasureString(writeValue + "_0.00", font).Height;
                gra.FillRectangle(textBackgroundBrush, x, y, textWidth * 0.8f, textHeight);
                gra.DrawString(writeValue, font, textColorBrush, new PointF(x, y));
                y += textHeight;
            }

            YoloHelper.RestoreMidCoordinates(data);
        }

        //画掩膜
        if (Config.ExecutionTaskMode == 2 || Config.ExecutionTaskMode == 3)
        {
            YoloHelper.RestoreDrawCoordinates(data);
            if (noMaskBcolor != null)
            {
                // 背景图
                Bitmap backImg = new Bitmap(imgData.Width, imgData.Height);

                // 背景绘制
                Graphics backDraw = Graphics.FromImage(backImg);

                backDraw.Clear((Color)noMaskBcolor);
                backDraw.Dispose();
                gra.DrawImage(backImg, PointF.Empty);
            }

            for (int i = 0; i < data.Count; i++)
            {
                // 矩形
                Rectangle rectangle = new Rectangle((int)data[i].BasicData[0], (int)data[i].BasicData[1],
                    (int)data[i].BasicData[2],
                    (int)data[i].BasicData[3]);

                // 颜色
                Color color = new Color();

                if (specifyLabelMaskColor == null)
                {
                    if (segmentationMaskRandomColor)
                    {
                        Random R = new Random();
                        color = Color.FromArgb(180, R.Next(0, 255), R.Next(0, 255), R.Next(0, 255));
                    }
                    else
                    {
                        color = Color.FromArgb(180, 0, 255, 0);
                    }
                }
                else
                {
                    if ((int)data[i].BasicData[5] + 1 > specifyLabelMaskColor.Length)
                    {
                        color = Color.FromArgb(180, 0, 255, 0);
                    }
                    else
                    {
                        color = specifyLabelMaskColor[(int)data[i].BasicData[5]];
                    }
                }

                Bitmap mask = YoloHelper.GenMaskImg_Memory(data[i].MaskData, color);
                gra.DrawImage(mask, rectangle);
            }

            YoloHelper.RestoreMidCoordinates(data);
        }

        if (Config.ExecutionTaskMode == 1 || Config.ExecutionTaskMode == 3 || Config.ExecutionTaskMode == 5)
        {
            YoloHelper.RestoreDrawCoordinates(data);
            for (int i = 0; i < data.Count; i++)
            {
                string confidenceDegree = data[i].BasicData[4].ToString("_0.00");
                if ((int)data[i].BasicData[5] + 1 > lableGroup.Length)
                {
                    writeValue = confidenceDegree;
                }
                else
                {
                    writeValue = lableGroup[(int)data[i].BasicData[5]] + confidenceDegree;
                }

                textWidth = gra.MeasureString(writeValue + "_0.00", font).Width;
                textHeight = gra.MeasureString(writeValue + "_0.00", font).Height;
                Rectangle rectangle = new Rectangle((int)data[i].BasicData[0], (int)data[i].BasicData[1],
                    (int)data[i].BasicData[2],
                    (int)data[i].BasicData[3]);
                gra.DrawRectangle(BorderBrush, rectangle);
                gra.FillRectangle(textBackgroundBrush, data[i].BasicData[0] - BorderBrush.Width / 2 - 1,
                    data[i].BasicData[1] - textHeight - BorderBrush.Width / 2 - 1, textWidth * 0.8f, textHeight);
                gra.DrawString(writeValue, font, textColorBrush, data[i].BasicData[0] - BorderBrush.Width / 2 - 1,
                    data[i].BasicData[1] - textHeight - BorderBrush.Width / 2 - 1);
            }

            YoloHelper.RestoreMidCoordinates(data);
        }

        if (Config.ExecutionTaskMode == 4 || Config.ExecutionTaskMode == 5)
        {
            YoloHelper.RestoreDrawCoordinates(data);
            if (data.Count > 0 && data[0].PointKeys.Length == 17)
            {
                // 颜色组
                Color[] colorGroup = new Color[]
                {
                    Color.Yellow,
                    Color.LawnGreen,
                    Color.LawnGreen,
                    Color.SpringGreen,
                    Color.SpringGreen,
                    Color.Blue,
                    Color.Blue,
                    Color.Firebrick,
                    Color.Firebrick,
                    Color.Firebrick,
                    Color.Firebrick,
                    Color.Blue,
                    Color.Blue,
                    Color.Orange,
                    Color.Orange,
                    Color.Orange,
                    Color.Orange
                };

                // 圆点半径
                int originPai = (imgData.Width > imgData.Height ? imgData.Height : imgData.Width) / 100;

                // 线条宽度
                int lineWidth = (imgData.Width > imgData.Height ? imgData.Height : imgData.Width) / 150;

                // 线条样式
                Pen lineStyle;

                for (int i = 0; i < data.Count; i++)
                {
                    lineStyle = new Pen(new SolidBrush(colorGroup[0]), lineWidth);

                    // 肩膀中心点
                    PointF shoulderMid = new PointF((data[i].PointKeys[5].X + data[i].PointKeys[6].X) / 2 + originPai,
                        (data[i].PointKeys[5].Y + data[i].PointKeys[6].Y) / 2 + originPai);

                    if (data[i].PointKeys[0].V > credibilityThresholdKeyPoints &&
                        data[i].PointKeys[5].V > credibilityThresholdKeyPoints &&
                        data[i].PointKeys[6].V > credibilityThresholdKeyPoints)
                        gra.DrawLine(lineStyle,
                            new PointF(data[i].PointKeys[0].X + originPai, data[i].PointKeys[0].Y + originPai),
                            shoulderMid);
                    lineStyle = new Pen(new SolidBrush(colorGroup[5]), lineWidth);
                    if (data[i].PointKeys[5].V > credibilityThresholdKeyPoints &&
                        data[i].PointKeys[6].V > credibilityThresholdKeyPoints)
                        gra.DrawLine(lineStyle,
                            new PointF(data[i].PointKeys[5].X + originPai, data[i].PointKeys[5].Y + originPai),
                            new PointF(data[i].PointKeys[6].X + originPai, data[i].PointKeys[6].Y + originPai));
                    if (data[i].PointKeys[11].V > credibilityThresholdKeyPoints &&
                        data[i].PointKeys[12].V > credibilityThresholdKeyPoints)
                        gra.DrawLine(lineStyle,
                            new PointF(data[i].PointKeys[11].X + originPai, data[i].PointKeys[11].Y + originPai),
                            new PointF(data[i].PointKeys[12].X + originPai, data[i].PointKeys[12].Y + originPai));
                    if (data[i].PointKeys[5].V > credibilityThresholdKeyPoints &&
                        data[i].PointKeys[11].V > credibilityThresholdKeyPoints)
                        gra.DrawLine(lineStyle,
                            new PointF(data[i].PointKeys[5].X + originPai, data[i].PointKeys[5].Y + originPai),
                            new PointF(data[i].PointKeys[11].X + originPai, data[i].PointKeys[11].Y + originPai));
                    if (data[i].PointKeys[6].V > credibilityThresholdKeyPoints &&
                        data[i].PointKeys[12].V > credibilityThresholdKeyPoints)
                        gra.DrawLine(lineStyle,
                            new PointF(data[i].PointKeys[6].X + originPai, data[i].PointKeys[6].Y + originPai),
                            new PointF(data[i].PointKeys[12].X + originPai, data[i].PointKeys[12].Y + originPai));
                    lineStyle = new Pen(new SolidBrush(colorGroup[0]), lineWidth);
                    if (data[i].PointKeys[0].V > credibilityThresholdKeyPoints &&
                        data[i].PointKeys[1].V > credibilityThresholdKeyPoints)
                        gra.DrawLine(lineStyle,
                            new PointF(data[i].PointKeys[0].X + originPai, data[i].PointKeys[0].Y + originPai),
                            new PointF(data[i].PointKeys[1].X + originPai, data[i].PointKeys[1].Y + originPai));
                    if (data[i].PointKeys[0].V > credibilityThresholdKeyPoints &&
                        data[i].PointKeys[2].V > credibilityThresholdKeyPoints)
                        gra.DrawLine(lineStyle,
                            new PointF(data[i].PointKeys[0].X + originPai, data[i].PointKeys[0].Y + originPai),
                            new PointF(data[i].PointKeys[2].X + originPai, data[i].PointKeys[2].Y + originPai));
                    lineStyle = new Pen(new SolidBrush(colorGroup[1]), lineWidth);
                    if (data[i].PointKeys[1].V > credibilityThresholdKeyPoints &&
                        data[i].PointKeys[3].V > credibilityThresholdKeyPoints)
                        gra.DrawLine(lineStyle,
                            new PointF(data[i].PointKeys[1].X + originPai, data[i].PointKeys[1].Y + originPai),
                            new PointF(data[i].PointKeys[3].X + originPai, data[i].PointKeys[3].Y + originPai));
                    if (data[i].PointKeys[2].V > credibilityThresholdKeyPoints &&
                        data[i].PointKeys[4].V > credibilityThresholdKeyPoints)
                        gra.DrawLine(lineStyle,
                            new PointF(data[i].PointKeys[2].X + originPai, data[i].PointKeys[2].Y + originPai),
                            new PointF(data[i].PointKeys[4].X + originPai, data[i].PointKeys[4].Y + originPai));
                    for (int j = 5; j < data[i].PointKeys.Length - 2; j++)
                    {
                        if (data[i].PointKeys[j].V > credibilityThresholdKeyPoints &&
                            data[i].PointKeys[j + 2].V > credibilityThresholdKeyPoints)
                        {
                            if (j != 9 && j != 10)
                            {
                                lineStyle = new Pen(new SolidBrush(colorGroup[j + 2]), lineWidth);
                                gra.DrawLine(lineStyle,
                                    new PointF(data[i].PointKeys[j].X + originPai, data[i].PointKeys[j].Y + originPai),
                                    new PointF(data[i].PointKeys[j + 2].X + originPai,
                                        data[i].PointKeys[j + 2].Y + originPai));
                            }
                        }
                    }

                    for (int j = 0; j < data[i].PointKeys.Length; j++)
                    {
                        if (data[i].PointKeys[j].V > credibilityThresholdKeyPoints)
                        {
                            // 位置
                            Rectangle location = new Rectangle((int)data[i].PointKeys[j].X, (int)data[i].PointKeys[j].Y,
                                originPai * 2,
                                originPai * 2);

                            gra.FillEllipse(new SolidBrush(colorGroup[j]), location);
                        }
                    }
                }
            }
            else if (data.Count > 0)
            {
                // 颜色组
                Color[] colorGroup = new Color[]
                {
                    Color.Yellow,
                    Color.Red,
                    Color.SpringGreen,
                    Color.Blue,
                    Color.Firebrick,
                    Color.Blue,
                    Color.Orange,
                    Color.Beige,
                    Color.LightGreen,
                    Color.DarkGreen,
                    Color.Magenta,
                    Color.White,
                    Color.OrangeRed,
                    Color.Orchid,
                    Color.PaleGoldenrod,
                    Color.PaleGreen,
                    Color.PaleTurquoise,
                    Color.PaleVioletRed,
                    Color.PaleGreen,
                    Color.PaleTurquoise,
                };

                // 圆点半径
                int originPai = (imgData.Width > imgData.Height ? imgData.Height : imgData.Width) / 100;

                foreach (var item in data)
                {
                    for (int i = 0; i < item.PointKeys.Length; i++)
                    {
                        if (item.PointKeys[i].V > credibilityThresholdKeyPoints)
                        {
                            Rectangle location = new Rectangle((int)item.PointKeys[i].X, (int)item.PointKeys[i].Y,
                                originPai * 2,
                                originPai * 2);
                            gra.FillEllipse(i > 20 ? new SolidBrush(Color.SaddleBrown) : new SolidBrush(colorGroup[i]),
                                location);
                        }
                    }
                }
            }

            YoloHelper.RestoreMidCoordinates(data);
        }
        else if (Config.ExecutionTaskMode == 6)
        {
            for (int i = 0; i < data.Count; i++)
            {
                string confidenceDegree = data[i].BasicData[4].ToString("_0.00");
                if ((int)data[i].BasicData[5] + 1 > lableGroup.Length)
                {
                    writeValue = confidenceDegree;
                }
                else
                {
                    writeValue = lableGroup[(int)data[i].BasicData[5]] + confidenceDegree;
                }

                textWidth = gra.MeasureString(writeValue + "_0.00", font).Width;
                textHeight = gra.MeasureString(writeValue + "_0.00", font).Height;

                // obb矩形结构
                OBBRectangularStructure obb = YoloHelper.OBBConversion(data[i]);

                PointF[] pf = { obb.pt1, obb.pt2, obb.pt3, obb.pt4, obb.pt1 };
                gra.DrawLines(BorderBrush, pf);

                // 右下角点
                PointF rightUpPointF = pf[0];

                foreach (var point in pf)
                {
                    if (point.X >= rightUpPointF.X && point.Y >= rightUpPointF.Y)
                    {
                        rightUpPointF = point;
                    }
                }

                gra.FillRectangle(textBackgroundBrush, rightUpPointF.X - BorderBrush.Width / 2 - 1,
                    rightUpPointF.Y + BorderBrush.Width / 2 - 1, textWidth * 0.8f,
                    textHeight);
                gra.DrawString(writeValue, font, textColorBrush, rightUpPointF.X - BorderBrush.Width / 2 - 1,
                    rightUpPointF.Y + BorderBrush.Width / 2 - 1);
            }
        }

        gra.Dispose();
        return reult;
    }
}