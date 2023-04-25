package newStart;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

/**
 * 第一个唯一数字
 */
public class FirstUnique {

    private HashSet<Integer> unique = new HashSet<>();
    private HashSet<Integer> repeated = new HashSet<>();
    private List<Integer> nums = new ArrayList<>();
    private int offset = 0;

    public FirstUnique(int[] nums) {
        for (int num : nums) {
            add(num);
        }
    }

    public int showFirstUnique() {
        if (unique.isEmpty()) {
            return -1;
        }
        for (int i = offset; i < nums.size(); i++) {
            if (unique.contains(nums.get(i))) {
                offset = i;
                return nums.get(i);
            }
        }
        return -1;
    }

    public void add(int value) {
        if (unique.contains(value)) {
            unique.remove(value);
            repeated.add(value);
        } else {
            if (!repeated.contains(value)) {
                unique.add(value);
                nums.add(value);
            }
        }
    }
}
