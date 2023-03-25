package newStart;

public class Solution {

    // 删除最短的子数组使剩余数组有序
    public int findLengthOfShortestSubarray(int[] arr) {
        int n = arr.length, right = n - 1;
        while (right > 0 && arr[right - 1] <= arr[right]) {
            right--;
        }
        if (right == 0) {
            return 0;
        }
        int ans = right;
        for (int left = 0; left == 0 || arr[left - 1] <= arr[left]; left++) {
            while (right < n && arr[right] < arr[left]) {
                right++;
            }
            ans = Math.min(ans, right - left - 1);
        }
        return ans;
    }

    // 气温变化趋势
    public int temperatureTrend(int[] temperatureA, int[] temperatureB) {
        int n = temperatureA.length;
        int ans = 0;
        int cur = 0;
        for (int i = 1; i < n; i++) {
            if (temperatureA[i - 1] < temperatureA[i] && temperatureB[i - 1] < temperatureB[i]) {
                cur++;
            } else if (temperatureA[i - 1] == temperatureA[i] && temperatureB[i - 1] == temperatureB[i]) {
                cur++;
            } else if (temperatureA[i - 1] > temperatureA[i] && temperatureB[i - 1] > temperatureB[i]) {
                cur++;
            } else {
                cur = 0;
            }
            ans = Math.max(ans, cur);
        }
        return ans;
    }
}
