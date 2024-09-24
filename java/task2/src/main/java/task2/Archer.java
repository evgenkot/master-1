package task2;

public class Archer extends Combatant {
    public Archer(String name) {
        super(name, 85, 12, 0.40);
    }

    @Override
    public void attack(Damageable target) {
        System.out.println(getName() + " shoots an arrow!");
        if (target instanceof Combatant && ((Combatant) target).dodge()) {
            System.out.println(target.getName() + " dodged arrow!");
        } else {
            target.takeDamage(getAttackPower());
            System.out.println(target.getName() + " was pierced by an arrow: " + getAttackPower() + ". HP left: " + target.getHealth());
        }
    }
}

