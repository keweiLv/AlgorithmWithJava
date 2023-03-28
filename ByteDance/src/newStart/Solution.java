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

    // 采购方案
    public int purchasePlans(int[] nums, int target) {
        int mod = 1000000007;
        int ans = 0;
        Arrays.sort(nums);
        int left = 0, right = nums.length - 1;
        while (left < right) {
            if (nums[left] + nums[right] > target) {
                right--;
            } else {
                ans += right - left;
                left++;
            }
            ans %= mod;
        }
        return ans % mod;
    }

    // 重复叠加字符串匹配
    public int repeatedStringMatch(String a, String b) {
        StringBuilder sb = new StringBuilder();
        int ans = 0;
        while (sb.length() < b.length() && ++ans > 0) {
            sb.append(a);
        }
        sb.append(a);
        int idx = sb.indexOf(b);
        if(idx == -1){
            return -1;
        }
        return idx + b.length() > a.length() * ans ? ans + 1 : ans;
    }

    // 找出字符串中第一个匹配项的下标
    public int strStr(String haystack, String needle) {
        int n = haystack.length(), m = needle.length();
        char[] s = haystack.toCharArray(), p = needle.toCharArray();
        for (int i = 0; i <= n - m; i++) {
            int a = i, b = 0;
            while (b < m && s[a] == p[b]) {
                a++;
                b++;
            }
            if (b == m) {
                return i;
            }
        }
        return -1;
    }

    // 统计只差一个字符的字串数目
    public int countSubstrings(String s, String t) {
        int ans = 0;
        int m = s.length(),n = t.length();
        for(int i = 0;;i<m;i++){
            for(int j = 0;j<n;j++){
                if(s.charAt(i) != t.charAt(j)){
                    int l = 0,r = 0;
                    while(i - l > 0 && j - l > 0 && s.charAt(i-l-1) == t.charAt(j-l-1)){
                        ++l;
                    }
                    while(i + r + 1 < m && j + r + 1 <n && s.charAt(i + r +1) == t.charAt(j + r +1)){
                        ++r;
                    }
                    ans += (1 + l) * (1 + r);
                }
            }
        }
        return ans;
    }

    // 最小展台数量
    public int minNumBooths(String[] demand) {
        int[] cnt = new int[26];
        int[] cur = new int[26];
        for (String s : demand) {
            Arrays.fill(cur, 0);
            for (int i = 0; i < s.length(); i++) {
                int id = (int) s.charAt(i) - 'a';
                cur[id]++;
            }
            for (int i = 0; i < 26; i++) {
                cnt[i] = Math.max(cnt[i], cur[i]);
            }
        }
        int ans = 0;
        for (int num : cnt) {
            ans += num;
        }
        return ans;
    }
}
