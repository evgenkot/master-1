package task2;

public class Mage extends Combatant {
    public Mage(String name) {
        super(name, 100, 15, 0.2);
    }

    @Override
    public void attack(Damageable target) {
        System.out.println(getName() + " uses magical spell!");
        if (target instanceof Combatant && ((Combatant) target).dodge()) {
            System.out.println(target.getName() + " dodged magical spell!");
        } else {
            target.takeDamage(getAttackPower());
            System.out.println(target.getName() + " takes magical damage: " + getAttackPower() + ". HP left: " + target.getHealth());
        }
    }
}

