package newStart;

import java.util.*;

public class Solution2 {

    // 两个非重叠子数组的最大和
    public int maxSumTwoNoOverlap(int[] nums, int firstLen, int secondLen) {
        int n = nums.length;
        int[] s = new int[n + 1];
        for (int i = 0; i < n; i++) {
            s[i + 1] = s[i] + nums[i];
        }
        return Math.max(f(s, firstLen, secondLen), f(s, secondLen, firstLen));
    }

    private int f(int[] s, int first, int sec) {
        int maxSumA = 0, res = 0;
        for (int i = first + sec; i < s.length; i++) {
            maxSumA = Math.max(maxSumA, s[i - sec] - s[i - sec - first]);
            res = Math.max(res, maxSumA + s[i] - s[i - sec]);
        }
        return res;
    }

    // 因子的组合
    public List<List<Integer>> getFactors(int n) {
        return dfs(2, n);
    }

    private List<List<Integer>> dfs(int start, int num) {
        if (num == 1) {
            return new ArrayList<>();
        }
        int qNum = (int) Math.sqrt(num);
        List<List<Integer>> result = new ArrayList<>();
        for (int i = start; i <= qNum; i++) {
            if (num % i == 0) {
                List<Integer> simple = new ArrayList<>();
                simple.add(i);
                simple.add(num / i);
                result.add(simple);
                List<List<Integer>> nexList = dfs(i, num / i);
                for (List<Integer> list : nexList) {
                    list.add(i);
                    result.add(list);
                }
            }
        }
        return result;
    }

    // 最长字符串链
    Map<String, Integer> ws = new HashMap<>();

    public int longestStrChain(String[] words) {
        for (String str : words) {
            ws.put(str, 0);
        }
        int ans = 0;
        for (String key : ws.keySet()) {
            ans = Math.max(ans, dfs(key));
        }
        return ans;
    }

    private int dfs(String s) {
        Integer cnt = ws.get(s);
        if (cnt > 0) {
            return cnt;
        }
        for (int i = 0; i < s.length(); i++) {
            String tmp = s.substring(0, i) + s.substring(i + 1);
            if (ws.containsKey(tmp)) {
                cnt = Math.max(cnt, dfs(tmp));
            }
        }
        ws.put(s, cnt + 1);
        return cnt + 1;
    }

    // 寻找二叉树的叶子节点
    public List<List<Integer>> findLeaves(TreeNode root) {
        List<List<Integer>> ans = new ArrayList<>();
        while (root != null) {
            List<Integer> list = new ArrayList<>();
            root = recur(root, list);
            ans.add(list);
        }
        return ans;
    }

    private TreeNode recur(TreeNode root, List<Integer> list) {
        if (root == null) {
            return null;
        }
        if (root.left == null && root.right == null) {
            list.add(root.val);
            return null;
        }
        root.left = recur(root.left, list);
        root.right = recur(root.right, list);
        return root;
    }

    // 检查一个数是否在数组中占绝大多数
    public boolean isMajorityElement(int[] nums, int target) {
        int n = nums.length;
        int left = 0, right = n - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return left + n / 2 < n && nums[left + n / 2] == target;
    }

    // 删除字符使频率相同
    public boolean equalFrequency(String word) {
        int[] set = new int[26];
        for (int i = 0; i < word.length(); i++) {
            set[word.charAt(i) - 'a']++;
        }
        for (int i = 0; i < 26; i++) {
            if (set[i] > 0) {
                set[i]--;
                int mark = 0;
                boolean ok = true;
                for (int num : set) {
                    if (num == 0) {
                        continue;
                    }
                    if (mark > 0 && num != mark) {
                        ok = false;
                        break;
                    }
                    mark = num;
                }
                if (ok) {
                    return true;
                }
                set[i]++;
            }
        }
        return false;
    }

    // 移动石子直到连续
    public int[] numMovesStones(int a, int b, int c) {
        int[] tmp = new int[]{a, b, c};
        Arrays.sort(tmp);
        a = tmp[0];
        b = tmp[1];
        c = tmp[2];
        return new int[]{
                c - a == 2 ? 0 : b - a <= 2 || c - b <= 2 ? 1 : 2, c - a - 2
        };
    }

    // 通知所有员工所需的时间
    public int numOfMinutes(int n, int headID, int[] manager, int[] informTime) {
        List<Integer> g[] = new ArrayList[n];
        Arrays.setAll(g, e -> new ArrayList<>());
        for (int i = 0; i < n; i++) {
            if (manager[i] >= 0) {
                g[manager[i]].add(i);
            }
        }
        return dfs(g, informTime, headID);
    }

    private int dfs(List<Integer>[] g, int[] informTime, int id) {
        int maxTime = 0;
        for (int num : g[id]) {
            maxTime = Math.max(maxTime, dfs(g, informTime, num));
        }
        return maxTime + informTime[id];
    }

    // 强整数
    public List<Integer> powerfulIntegers(int x, int y, int bound) {
        Set<Integer> set = new HashSet<>();
        for (int a = 1; a <= bound; a *= x) {
            for (int b = 1; a + b <= bound; b *= y) {
                set.add(a + b);
                if (y == 1) {
                    break;
                }
            }
            if (x == 1) {
                break;
            }
        }
        return new ArrayList<>(set);
    }

    // 检查替换后的词是否有效
    public static boolean isValid(String s) {
        String tmp = s;
        while (!tmp.isBlank()) {
            String replace = tmp.replace("abc", "");
            if (replace.equals(tmp)) {
                return false;
            }
            tmp = replace;
        }
        return true;
    }

    // 摘水果
    public int maxTotalFruits(int[][] fruits, int startPos, int k) {
        int left = lowerBound(fruits, startPos - k);
        int right = left, s = 0, n = fruits.length;
        for (; right < n && fruits[right][0] <= startPos; right++) {
            s += fruits[right][1];
        }
        int ans = s;
        for (; right < n && fruits[right][0] <= startPos + k; right++) {
            s += fruits[right][1];
            while (fruits[right][0] * 2 - fruits[left][0] - startPos > k && fruits[right][0] - fruits[left][0] * 2 + startPos > k) {
                s -= fruits[left++][1];
            }
            ans = Math.max(ans, s);
        }
        return ans;
    }

    private int lowerBound(int[][] fruits, int target) {
        int left = -1, right = fruits.length;
        while (left + 1 < right) {
            int mid = left + (right - left) / 2;
            if (fruits[mid][0] < target) {
                left = mid;
            } else {
                right = mid;
            }
        }
        return right;
    }

    // 有序数组中的缺失元素
    public int missingElement(int[] nums, int k) {
        int lose = nums[nums.length - 1] - nums[0] + 1 - nums.length;
        if (k > lose) {
            return nums[nums.length - 1] + k - lose;
        }
        int l = 0;
        int r = nums.length - 1;
        while (l < r - 1) {
            int mid = l + (r - l) >> 1;
            int loseCnt = nums[mid] - nums[l] - (mid - l);
            if (k > loseCnt) {
                l = mid;
                k -= loseCnt;
            } else {
                r = mid;
            }
        }
        return nums[l] + k;
    }

    // 处理用时最长的那个任务的员工
    public int hardestWorker(int n, int[][] logs) {
        int ans = 0;
        int last = 0, mx = 0;
        for (int[] log : logs) {
            int uid = log[0], time = log[1];
            time -= last;
            if (time > mx || (time == mx && uid < ans)) {
                ans = uid;
                mx = time;
            }
            last += time;
        }
        return ans;
    }

}
