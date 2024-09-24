package task2;

public class App {
    public static void main(String[] args) {
        System.out.println("Hello World!");
        ChildClass mySuper=new ChildClass("Evgeny", 150, 160, 170);
        System.out.println("mySuper name is " + mySuper.GetName() + " " + mySuper.Greeting());
        mySuper.UseSkills();
        System.out.println(mySuper.magic);
    }
}
