using OpenCvSharp;

namespace CanCanNeedJJBong.Yolo.Core;

/// <summary>
/// Yolo的数据
/// </summary>
public class YoloData
{
    /// <summary>
    /// 基本数据
    /// 用于存放6个基本数据:中心x,中心y,宽,高,置信度,标签索引；如果是obb模型会返回7个数据,最后是旋转角度r；如使用了分类模型,则只会返回2个数据,分别代表置信度和标签索引
    /// </summary>
    public float[] BasicData { get; set; }

    /// <summary>
    /// 掩膜数据
    /// 初期用于存放分割模型的32个mask数据,在后处理中,会对该值重新赋予一个还原后的掩膜数据。它是由0和1组成的矩阵，1代表掩膜部分、掩膜的起始坐标就是对应目标检测框的左上角,你可以理解成,这个掩膜跟检测框一样大,你通过检测框的坐标,就知道掩膜的坐标了
    /// </summary>
    public Mat MaskData { get; set; }

    /// <summary>
    /// 关键点
    /// 使用pose模型的关键点信息
    /// </summary>
    public Pose[] PointKeys { get; set; }
}