package LC_RACE.WeeklyRace_344;

import java.util.*;

public class Solution {

    // 找出不同元素数目差数组
    public int[] distinctDifferenceArray(int[] nums) {
        Set<Integer> pre = new HashSet<>();
        Map<Integer, Integer> tail = new HashMap<>();
        int n = nums.length;
        int[] ans = new int[n];
        for (int num : nums) {
            tail.put(num, tail.getOrDefault(num, 0) + 1);
        }
        for (int i = 0; i < n; i++) {
            pre.add(nums[i]);
            Integer cnt = tail.get(nums[i]);
            if (cnt == 1){
                tail.remove(nums[i]);
            }else {
                tail.put(nums[i],tail.get(nums[i]) - 1);
            }
            ans[i] = pre.size() - tail.size();
        }
        return ans;
    }
}
