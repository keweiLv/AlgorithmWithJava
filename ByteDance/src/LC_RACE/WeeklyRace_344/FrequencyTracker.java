package LC_RACE.WeeklyRace_344;

import java.util.HashMap;
import java.util.Map;

/**
 * 频率跟踪器
 */
public class FrequencyTracker {

    // 记录每个数及其出现的次数
    Map<Integer, Integer> cnt = new HashMap<>();
    // 记录出现次数的出现次数
    Map<Integer, Integer> frequencyMap = new HashMap<>();

    public FrequencyTracker() {

    }

    public void add(int number) {
        frequencyMap.merge(cnt.getOrDefault(number, 0), -1, Integer::sum);
        int c = cnt.merge(number, 1, Integer::sum);
        frequencyMap.merge(c, 1, Integer::sum);
    }

    public void deleteOne(int number) {
        if (cnt.getOrDefault(number, 0) == 0) {
            return;
        }
        frequencyMap.merge(cnt.get(number), -1, Integer::sum);
        Integer merge = cnt.merge(number, -1, Integer::sum);
        frequencyMap.merge(merge, 1, Integer::sum);
    }

    public boolean hasFrequency(int frequency) {
        return frequencyMap.getOrDefault(frequency, 0) > 0;
    }
}
