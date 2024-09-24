package task2;

class ChildClass extends SuperClass implements MyInterface
{
    int magic;

    public ChildClass() {
        super();
    }

    public ChildClass(String name, int energy, int health, int magic) {
        super(name, energy, health);
        this.magic=magic;
    }
    
    @Override
    public String Greeting() {
        return "Reply from ChildClass";
    }

    @Override
    public void UseSkills()
    {
        if(magic<10) magic=0;
        else magic-=10;
    }
}
