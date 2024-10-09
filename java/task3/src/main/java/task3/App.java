package task3;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class App {
    public static void main(String[] args) {
        // Проверяем, передан ли путь к файлу в аргументах
        String filePath = args.length > 0 ? args[0] : "text.txt"; // Используем text.txt по умолчанию

        HashMap<String, Integer> wordCount = new HashMap<>();

        HashSet<String> blackList = new HashSet<>(Arrays.asList(
            "и", "в", "не", "на", "с", "что", "как", "по", "за", "то", 
            "это", "а", "но", "или", "так", "также", "к", "для", "от", 
            "до", "если", "когда", "чтобы", "все", "всё", "да", "нет",
            "я", "ты", "мне", "он", "из", "его", "она", "меня", "о"
        ));

        // Чтение файла и подсчет слов
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                // Разделение строки на слова, удаление знаков препинания и приведение к нижнему регистру
                String[] words = line.replaceAll("[^a-zA-Zа-яА-Я0-9 ]", "").toLowerCase().split("\\s+");
                for (String word : words) {
                    if (!word.isEmpty() && !blackList.contains(word)) {
                        wordCount.put(word, wordCount.getOrDefault(word, 0) + 1);
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Переносим записи в список для сортировки
        ArrayList<HashMap.Entry<String, Integer>> wordList = new ArrayList<>(wordCount.entrySet());

        // Сортировка по частоте 
        Collections.sort(wordList, (a, b) -> a.getValue().compareTo(b.getValue()));

        // Вывод топ-20 слов
        System.out.println("Top 20 words:");
        for (int i = 0; i < 20 && i < wordList.size(); i++) {
            HashMap.Entry<String, Integer> entry = wordList.get(i);
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
    }
}

