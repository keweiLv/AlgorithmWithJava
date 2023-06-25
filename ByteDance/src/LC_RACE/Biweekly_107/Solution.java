package LC_RACE.Biweekly_107;

/**
 * 第 107 场双周赛
 */
public class Solution {

    // 最大字符串配对数目
    public int maximumNumberOfStringPairs(String[] words) {
        int n = words.length;
        int[] vis = new int[n];
        int ans = 0;
        for (int i = 0; i < n; i++) {
            String t1 = "";
            if (vis[i] == 1) {
                continue;
            }
            t1 = words[i];
            for (int j = i + 1; j < n; j++) {
                StringBuilder t2 = new StringBuilder();
                if (vis[j] == 1) {
                    continue;
                }
                t2 = new StringBuilder(words[j]).reverse();
                if (t1.contentEquals(t2)) {
                    ans++;
                    vis[i] = 1;
                    vis[j] = 1;
                }
            }
        }
        return ans;
    }

    // 构造最长的子字符串
    public int longestString(int x, int y, int z) {
        return (Math.min(x, y) * 2 + (x != y ? 1 : 0) + z) * 2;
    }

    // 字符串连接删减字母
    public int minimizeConcatenatedLength(String[] words) {
        return 0;
    }

}
