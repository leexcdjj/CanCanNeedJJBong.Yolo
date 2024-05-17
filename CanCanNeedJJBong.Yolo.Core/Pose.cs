namespace CanCanNeedJJBong.Yolo.Core;

/// <summary>
/// Post模型的点数据
/// </summary>
public class Pose
{
    /// <summary>
    /// pose点的x坐标
    /// </summary>
    public float X { get; set; }

    /// <summary>
    /// post点的y坐标
    /// </summary>
    public float Y { get; set; }

    /// <summary>
    /// pose点的可信度,当该值较低时,实际大概率是在框外面,一般以0.5为阈值
    /// </summary>
    public float V { get; set; }
}