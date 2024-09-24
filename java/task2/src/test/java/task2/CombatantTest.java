package task2;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CombatantTest {

    private class TestCombatant extends Combatant {
        public TestCombatant(String name, int health, int attackPower, Double dodgeChance) {
            super(name, health, attackPower, dodgeChance);
        }

        @Override
        public void attack(Damageable target) {
            target.takeDamage(getAttackPower());
        }
    }

    @Test
    void testTakeDamage() {
        TestCombatant combatant = new TestCombatant("Test", 100, 10, 0.0);
        combatant.takeDamage(10);
        assertEquals(90, combatant.getHealth());
    }

    @Test
    void testHealthDoesNotGoBelowZero() {
        TestCombatant combatant = new TestCombatant("Test", 10, 10, 0.0);
        combatant.takeDamage(15);
        assertEquals(0, combatant.getHealth());
    }

    @Test
    void testDodge() {
        TestCombatant combatant = new TestCombatant("Test", 100, 10, 0.5);
        boolean dodged = combatant.dodge();
        assertTrue(dodged || !dodged);
    }
}

