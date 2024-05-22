using System.Drawing;
using CanCanNeedJJBong.Yolo.Core;

namespace CanCanNeedJJBong.Yolo.TestConsole;

class Program
{
    /// <summary>
    /// 置信度
    /// </summary>
    private static float confidenceDegree => 0.5f;

    /// <summary>
    /// 全局iou
    /// </summary>
    private static bool allIou => false;

    /// <summary>
    /// iou阈值
    /// </summary>
    private static float iou = float.Parse("0.3");

    static void Main(string[] args)
    {
        Console.WriteLine("Hello, Yolo!");
        Console.WriteLine("请输入推理测试模式 => 0:分类,1:检测,2:分割,3:检测加分割,4:关键点,5:检测加关键点,6:OBB检测");

        if (!int.TryParse(Console.ReadLine(), out int taskMode))
        {
            Console.WriteLine("输入无效");
            return;
        }

        YoloTask task = taskMode switch
        {
            0 => new ClsTask(),
            1 => new DetectionTask(),
            2 or 3 => new SegTask(),
            4 or 5 => new KeyPointsTask(),
            6 => new OBBTask(),
            _ => throw new Exception("未知模式"),
        };

        task.Execute();

        Console.ReadLine();
    }

    public static void Cls()
    {
        ReasonAndGen("螺丝刀2.jpg", "yolov8n-cls.onnx", "cls", 0);
    }

    public static void Detection()
    {
        ReasonAndGen("OBB测试飞机场2.bmp", "yolov9-c.onnx", "Detection", 1);
    }

    public static void Seg()
    {
        ReasonAndGen("83c6a5691f1464f9c0e963f5d42bf3e0.jpeg", "yolov8n-seg.transd.onnx", "Seg", 3);
    }

    public static void keyPoints()
    {
        ReasonAndGen("4.bmp", "yolov8n-pose.transd.onnx", "KeyPoints", 5);
    }

    public static void OBB()
    {
        ReasonAndGen("OBB测试球场.jpg", "yolov8n-obb.transd.onnx", "OBB", 6);
    }

    public static void ReasonAndGen(string testImgName, string modelName, string action, int taskMode)
    {
        string currentDirectory = Directory.GetCurrentDirectory();

        string modelsDirectory = Path.Combine(currentDirectory, "Models");
        string testImgDirectory = Path.Combine(currentDirectory, "TestImg");
        string genImgDirectory = Path.Combine(currentDirectory, "GenImg");

        if (!Directory.Exists(genImgDirectory))
        {
            Directory.CreateDirectory(genImgDirectory);
        }

        string genImgName = testImgName.Replace(Path.GetExtension(testImgName), "") + action + ".jpg";

        string modelPath =
            Path.Combine(modelsDirectory, modelName);

        string testImgPath =
            Path.Combine(testImgDirectory,
                testImgName);

        string genImgPath =
            Path.Combine(genImgDirectory,
                genImgName);

        var yolo = new YoloService(modelPath, false, 0, 8, taskMode);

        Bitmap imgBit = new Bitmap(testImgPath);
        var data = yolo.ModelReasoning(imgBit, confidenceDegree, iou, allIou, 1);

        var img = yolo.GenerateIma(imgBit, data);

        img.Save(genImgPath);
    }
}