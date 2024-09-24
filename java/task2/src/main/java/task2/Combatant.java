package task2;

import java.util.Random;

public abstract class Combatant implements Attackable, Damageable {
    private String name;
    private int health;
    private int attackPower;
    private Double dodgeChance;

    public Combatant(String name, int health, int attackPower, Double dodgeChance) {
        this.name = name;
        this.health = health;
        this.attackPower = attackPower;
        this.dodgeChance = dodgeChance;
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public int getHealth() {
        return health;
    }

    @Override
    public void takeDamage(int damage) {
        this.health -= damage;
        if (this.health < 0) this.health = 0;
    }

    public int getAttackPower() {
        return attackPower;
    }

    public boolean dodge() {
        Random random = new Random();
        return random.nextDouble() < dodgeChance;
    }

    @Override
    public abstract void attack(Damageable target);
}

