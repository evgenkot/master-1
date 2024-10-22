package task4;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class Calculator extends JFrame implements ActionListener {
    private JTextField inputField;
    private JButton[] numberButtons;
    private JButton addButton, subButton, mulButton, divButton, powButton, equalButton, clearButton, decimalButton, backspaceButton, piButton;
    private double num1, num2, result;
    private char operator;

    public Calculator() {
        setTitle("Swing Calc");
        setSize(400, 600);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());

        inputField = new JTextField();
        inputField.setEditable(false);
        inputField.setFont(new Font("Arial", Font.PLAIN, 24));
        inputField.setBackground(Color.DARK_GRAY);
        inputField.setForeground(Color.WHITE);
        add(inputField, BorderLayout.NORTH);

        JPanel buttonPanel = new JPanel();
        buttonPanel.setLayout(new GridLayout(4, 5, 10, 10));
        buttonPanel.setBackground(Color.DARK_GRAY);

        numberButtons = new JButton[10];
        for (int i = 0; i < 10; i++) {
            numberButtons[i] = new JButton(String.valueOf(i));
            numberButtons[i].setFont(new Font("Arial", Font.PLAIN, 24));
            numberButtons[i].setBackground(Color.GRAY);
            numberButtons[i].setForeground(Color.WHITE);
            numberButtons[i].addActionListener(this);
        }

        addButton = new JButton("+");
        subButton = new JButton("-");
        mulButton = new JButton("*");
        divButton = new JButton("/");
        powButton = new JButton("^");
        equalButton = new JButton("=");
        clearButton = new JButton("C");
        decimalButton = new JButton(".");
        backspaceButton = new JButton("←");
        piButton = new JButton("π");

        JButton[] operationButtons = {addButton, subButton, mulButton, divButton, powButton, equalButton, clearButton, decimalButton, backspaceButton, piButton};
        for (JButton button : operationButtons) {
            button.setFont(new Font("Arial", Font.PLAIN, 24));
            button.setBackground(Color.GRAY);
            button.setForeground(Color.WHITE);
            button.addActionListener(this);
        }

        buttonPanel.add(numberButtons[7]);
        buttonPanel.add(numberButtons[8]);
        buttonPanel.add(numberButtons[9]);
        buttonPanel.add(backspaceButton);
        buttonPanel.add(clearButton);

        buttonPanel.add(numberButtons[4]);
        buttonPanel.add(numberButtons[5]);
        buttonPanel.add(numberButtons[6]);
        buttonPanel.add(addButton);
        buttonPanel.add(subButton);

        buttonPanel.add(numberButtons[1]);
        buttonPanel.add(numberButtons[2]);
        buttonPanel.add(numberButtons[3]);
        buttonPanel.add(mulButton);
        buttonPanel.add(divButton);

        buttonPanel.add(numberButtons[0]);
        buttonPanel.add(decimalButton);
        buttonPanel.add(piButton);
        buttonPanel.add(powButton);
        buttonPanel.add(equalButton);

        add(buttonPanel, BorderLayout.CENTER);
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        for (int i = 0; i < 10; i++) {
            if (e.getSource() == numberButtons[i]) {
                inputField.setText(inputField.getText() + i);
            }
        }

        if (e.getSource() == decimalButton) {
            if (!inputField.getText().contains(".")) {
                inputField.setText(inputField.getText() + ".");
            }
        }

        if (e.getSource() == addButton) {
            performOperation('+');
        } else if (e.getSource() == subButton) {
            performOperation('-');
        } else if (e.getSource() == mulButton) {
            performOperation('*');
        } else if (e.getSource() == divButton) {
            performOperation('/');
        } else if (e.getSource() == powButton) {
            performOperation('^');
        } else if (e.getSource() == equalButton) {
            calculateResult();
        } else if (e.getSource() == clearButton) {
            inputField.setText("");
        } else if (e.getSource() == backspaceButton) {
            String currentText = inputField.getText();
            if (currentText.length() > 0) {
                inputField.setText(currentText.substring(0, currentText.length() - 1));
            }
        } else if (e.getSource() == piButton) {
            inputField.setText(inputField.getText() + Math.PI);
        }
    }

    private void performOperation(char op) {
        try {
            num1 = Double.parseDouble(inputField.getText());
            operator = op;
            inputField.setText("");
        } catch (NumberFormatException e) {
            inputField.setText("Input error");
        }
    }

    private void calculateResult() {
        try {
            num2 = Double.parseDouble(inputField.getText());
            switch (operator) {
                case '+':
                result = num1 + num2;
                break;
                case '-':
                result = num1 - num2;
                break;
                case '*':
                result = num1 * num2;
                break;
                case '/':
                if (num2 != 0) {
                    result = num1 / num2;
                } else {
                    inputField.setText("Division by 0");
                    return;
                }
                break;
                case '^':
                result = Math.pow(num1, num2);
                break;
            }
            inputField.setText(String.valueOf(result));
        } catch (NumberFormatException e) {
            inputField.setText("Input Error");
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            Calculator calculator = new Calculator();
            calculator.setVisible(true);
        });
    }
}

