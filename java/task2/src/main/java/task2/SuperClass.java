package task2;

interface MyInterface 
{
    public void UseSkills();
}

class SuperClass
{
    protected String name;
    protected int energy;
    protected int health;

    public SuperClass()
    {
        name="";
        energy=0;
    }

    public SuperClass(String name, int energy, int health)
    {
        this.name=name;
        this.energy=energy;
    }

    public String GetName()
    {
        return this.name;
    }

    public void SetName(String name)
    {
        this.name=name;
    }
    public String Greeting()
    {
        return "Default greeting";
    }
}

