package start240304;

import java.util.HashMap;
import java.util.Map;

public class FrequencyTracker {

    private final Map<Integer, Integer> cnt = new HashMap<>();
    private final Map<Integer, Integer> freq = new HashMap<>();


    public FrequencyTracker() {

    }

    public void add(int number) {
        update(number, 1);
    }

    public void deleteOne(int number) {
        if (cnt.getOrDefault(number, 0) > 0) {
            update(number, -1);
        }
    }

    public boolean hasFrequency(int frequency) {
        return freq.getOrDefault(frequency, 0) > 0;
    }

    public void update(int number, int val) {
        int c = cnt.merge(number, val, Integer::sum);
        freq.merge(c - val, -1, Integer::sum);
        freq.merge(c, 1, Integer::sum);
    }
}
