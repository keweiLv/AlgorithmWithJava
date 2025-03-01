package again250224;

import javafx.util.Pair;

import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeSet;

/**
 * 设计食物评分系统
 */
public class FoodRatings {

    private Map<String, TreeSet<Pair<Integer, String>>> cuisineMap = new HashMap<>();
    private Map<String, Pair<Integer, String>> foodMap = new HashMap<>();
    private final Comparator<Pair<Integer,String>> cmp = (a,b) ->{
        if (!a.getKey().equals(b.getKey())) {
            return b.getKey() - a.getKey();
        }
        return a.getValue().compareTo(b.getValue());
    };

    public FoodRatings(String[] foods, String[] cuisines, int[] ratings) {
        for(int i = 0;i<foods.length;i++){
            String food = foods[i],cuisine = cuisines[i];
            int rating = ratings[i];
            cuisineMap.computeIfAbsent(cuisine, k -> new TreeSet<>(cmp)).add(new Pair<>(rating,food));
            foodMap.put(food,new Pair<>(rating,cuisine));
        }
    }

    public void changeRating(String food, int newRating) {
        Pair<Integer,String> old = foodMap.get(food);
        int oldRating = old.getKey();
        String cuision = old.getValue();
        foodMap.put(food, new Pair<Integer,String>(newRating, cuision));
        cuisineMap.get(cuision).remove(new Pair<>(oldRating,food));
        cuisineMap.get(cuision).add(new Pair<>(newRating,food));
    }

    public String highestRated(String cuisine) {
        return cuisineMap.get(cuisine).first().getValue();
    }

}
