import json
import os
filename="students.json"
print("文件将保存在：",os.path.abspath(filename))
if os.path.exists(filename):
    with open(filename,"r",encoding="utf-8") as f:
        dic=json.load(f)
else:
    dic={}

def save_data():
    with open(filename,"w",encoding="utf-8") as f:
        json.dump(dic,f,ensure_ascii=False,indent=4)

def search_student():
    num=input("请输入要查询的学生学号:")
    if num in dic:
        print("查询结果：","\t".join(str(x) for x in dic[num]))
    else:
        print("查无此人")

def add_student():
    num=input("请输入学生学号:")
    if num in dic:
        print("该学号已存在，是否覆盖？(y/n)")
        choice=input()
        if choice.lower()!='y':
            print("取消录入")
            return
    name=input("请输入学生姓名:")
    gnd=input("请输入学生性别:")
    rnum=input("请输入学生宿舍房间号:")
    phnum=input("请输入学生联系电话:")
    dic[num]=[num,name,gnd,rnum,phnum]
    print("录入成功")

def show_all():
    if not dic:
        print("当前没有学生信息")
        return
    print("学号\t姓名\t性别\t宿舍房间号\t联系电话")
    for i in sorted(dic.keys(),key=lambda x:int(x)):
        print("\t".join(str(x) for x in dic[i]))
    print(f"总人数：{len(dic)}")

def delete_student():
    num=input("请输入要删除的学生学号:")
    if num in dic:
        print("找到该学生：", "\t".join(str(x) for x in dic[num]))
        confirm=input("确认删除？(y/n):")
        if confirm.lower()=="y":
            del dic[num]
            print("删除成功")
        else:
            print("已取消删除")
    else:
        print("查无此人")

def export_to_txt():
    out_file="students.txt"
    with open(out_file,"w",encoding="utf-8") as f:
        f.write("学号\t姓名\t性别\t宿舍房间号\t联系电话\n")
        for i in sorted(dic.keys(),key=lambda x:int(x)):
            f.write("\t".join(str(x) for x in dic[i])+"\n")
    print(f"信息已导出到 {os.path.abspath(out_file)}")

def statistics():
    total=len(dic)
    male=sum(1 for i in dic if dic[i][2]=="男")
    female=sum(1 for i in dic if dic[i][2]=="女")
    print(f"总人数: {total},男生: {male},女生: {female}")

while True:
    print("\n======= 学生宿舍管理程序 =======")
    print("1: 按学号查找学生信息")
    print("2: 录入新的学生信息")
    print("3: 显示所有学生信息")
    print("4: 删除学生信息")
    print("5: 导出所有学生信息到文本文件")
    print("6: 统计学生人数与性别比例")
    print("7: 保存并退出")
    print("================================")
    choice=input("请输入功能选项(1-7): ")
    if choice == '1':
        search_student()
    elif choice == '2':
        add_student()
    elif choice == '3':
        show_all()
    elif choice == '4':
        delete_student()
    elif choice == '5':
        export_to_txt()
    elif choice == '6':
        statistics()
    elif choice == '7':
        save_data()
        print("数据已保存，程序结束。")
        break
    else:
        print("输入错误，请重新输入")
