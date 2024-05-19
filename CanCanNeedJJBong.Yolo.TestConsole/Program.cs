using System.Drawing;
using CanCanNeedJJBong.Yolo.Core;

namespace CanCanNeedJJBong.Yolo.TestConsole;

class Program
{
    /// <summary>
    /// 置信度
    /// </summary>
    private static float confidenceDegree => float.Parse("0.5");

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
        Console.WriteLine("请输入推理测试模式 => 1:分类,2:检测,3:分割,4:检测加分割,5:关键点,6:检测加关键点,7:OBB检测");
        
        int taskMode = int.Parse(Console.ReadLine());
        
        // cls测试
        switch (taskMode)
        {
            case 1:
                Cls();
                break;
            case 2:
                // 错误
                Detection();
                break;
            case 3:
            case 4:
                // 错误
                Seg();
                break;
            case 5:
            case 6:
                keyPoints();
                break;
            case 7:
                OBB();
                break;
            default:
                throw new Exception("未知");
        }

        Console.WriteLine("推理生成完成");
        Console.ReadLine();

    }

    public static void Cls()
    {
        ReasonAndGen("螺丝刀2.jpg","yolov8n-cls.onnx","cls");
    }
    
    public static void Detection()
    {
        ReasonAndGen("OBB测试飞机场2.bmp","yolov9-c.onnx","Detection");
    }
    
    public static void Seg()
    {
        ReasonAndGen("83c6a5691f1464f9c0e963f5d42bf3e0.jpeg","yolov8n-seg.transd.onnx","Seg");
    }

    public static void keyPoints()
    {
        ReasonAndGen("4.bmp", "yolov8n-pose.transd.onnx","KeyPoints");
    }

    public static void OBB()
    {
        ReasonAndGen("OBB测试球场.jpg", "yolov8n-obb.transd.onnx","OBB");
    }

    public static void ReasonAndGen(string testImgName,string modelName,string action)
    {
        string genImgName = testImgName.Replace(Path.GetExtension(testImgName),"")  + action+".jpg";
        
        string modelPath =
            Path.Combine("D:\\Source\\My\\CanCanNeedJJBong.Yolo\\CanCanNeedJJBong.Yolo.TestConsole\\Models", modelName);
        
        string testImgPath =
            Path.Combine("D:\\Source\\My\\CanCanNeedJJBong.Yolo\\CanCanNeedJJBong.Yolo.TestConsole\\TestImg",
                testImgName);
        
        string genImgPath =
            Path.Combine("D:\\Source\\My\\CanCanNeedJJBong.Yolo\\CanCanNeedJJBong.Yolo.TestConsole\\GenImg",
                genImgName);

        var yolo = new YoloService(modelPath, false, 0, 8);
        
        Bitmap imgBit = new Bitmap(testImgPath);
        var data = yolo.ModelReasoning(imgBit, confidenceDegree, iou, allIou, 1);

        var img = yolo.GenerateIma(imgBit, data);
        
        img.Save(genImgPath);
        
    }
    
}