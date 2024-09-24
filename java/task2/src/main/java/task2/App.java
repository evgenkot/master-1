package task2;

public class App {
    public static void main(String[] args) {
        System.out.println("Hello World!");
        Combatant mage = new Mage("Garry");
        Combatant archer = new Archer("Legolaz");

        Battle battle = new Battle();
        battle.startBattle(mage, archer);
    }
}
