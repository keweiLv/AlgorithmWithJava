package newStart;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * O（1）时间插入、删除和获取随机元素
 */
public class RandomizedSet {

    int[] nums = new int[200010];
    Map<Integer, Integer> map = new HashMap<>();
    Random random = new Random();
    int idx = -1;

    public RandomizedSet() {

    }

    public boolean insert(int val) {
        if (map.containsKey(val)) {
            return false;
        }
        nums[++idx] = val;
        map.put(val, idx);
        return true;
    }

    public boolean remove(int val) {
        if (!map.containsKey(val)) {
            return false;
        }
        Integer remove = map.remove(val);
        if (remove != idx) {
            map.put(nums[idx], remove);
        }
        nums[remove] = nums[idx--];
        return true;
    }

    public int getRandom() {
        return nums[random.nextInt(idx + 1)];
    }
}
