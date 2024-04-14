package start240304;

import java.util.*;

public class Solution {

    // 最短单词距离
    public int shortestDistance(String[] wordsDict, String word1, String word2) {
        int len = wordsDict.length;
        int ans = len;
        int idx1 = -1, idx2 = -1;
        for (int i = 0; i < len; i++) {
            String word = wordsDict[i];
            if (word.equals(word1)) {
                idx1 = i;
            } else if (word.equals(word2)) {
                idx2 = i;
            }
            if (idx1 >= 0 && idx2 >= 0) {
                ans = Math.min(ans, Math.abs(idx1 - idx2));
            }
        }
        return ans;
    }

    // 三角形最小路径和
    public int minimumTotal(List<List<Integer>> triangle) {
        int n = triangle.size();
        int[] dp = new int[n + 1];
        for (int i = n - 1; i >= 0; i--) {
            for (int j = 0; j <= i; j++) {
                dp[j] = Math.min(dp[j], dp[j + 1]) + triangle.get(i).get(j);
            }
        }
        return dp[0];
    }

    // 无重复字符的最长子串
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> map = new HashMap<>();
        int max = 0, left = 0;
        for (int i = 0; i < s.length(); i++) {
            if (map.containsKey(s.charAt(i))) {
                left = Math.max(left, map.get(s.charAt(i)) + 1);
            }
            map.put(s.charAt(i), i);
            max = Math.max(max, i - left + 1);
        }
        return max;
    }

    // 找出美丽数组的最小和
    final int MOD = (int) 1e9 + 7;

    public int minimumPossibleSum(int n, int target) {
        int m = target / 2;
        if (n <= m) {
            return (int) ((long) (1 + n) * n / 2 % MOD);
        }
        return (int) (((long) (1 + m) * m / 2 + ((long) target + target + (n - m) - 1) * (n - m) / 2) % MOD);
    }

    // 猜数字游戏
    public String getHint(String secret, String guess) {
        int bulls = 0;
        int[] cntS = new int[10];
        int[] cntG = new int[10];
        for (int i = 0; i < guess.length(); i++) {
            if (secret.charAt(i) == guess.charAt(i)) {
                bulls++;
            } else {
                cntS[secret.charAt(i) - '0']++;
                cntG[guess.charAt(i) - '0']++;
            }
        }
        int cows = 0;
        for (int i = 0; i < 10; i++) {
            cows += Math.min(cntS[i], cntG[i]);
        }
        return Integer.toString(bulls) + "A" + Integer.toString(cows) + "B";
    }

    // 二叉树的层平均值
    public List<Double> averageOfLevels(TreeNode root) {
        List<Double> averages = new ArrayList<>();
        Queue<TreeNode> queue = new ArrayDeque<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            double sum = 0;
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode poll = queue.poll();
                sum += poll.val;
                TreeNode left = poll.left;
                TreeNode right = poll.right;
                if (left != null) {
                    queue.offer(left);
                }
                if (right != null) {
                    queue.offer(right);
                }
            }
            averages.add(sum / size);
        }
        return averages;
    }

    // 最大二进制奇数
    public String maximumOddBinaryNumber(String s) {
        int cnt = (int) s.chars().filter(c -> c == '1').count();
        return "1".repeat(cnt - 1) + "0".repeat(s.length() - cnt) + "1";
    }

    // 合并后数组中的最大元素
    public long maxArrayValue(int[] nums) {
        int n = nums.length;
        long sum = nums[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            sum = nums[i] <= sum ? sum + nums[i] : nums[i];
        }
        return sum;
    }

    // 好子数组的最大分数
    public int maximumScore(int[] nums, int k) {
        int n = nums.length;
        int ans = nums[k];
        int min = nums[k];
        int left = k;
        int right = k;
        while (left > 0 || right + 1 < n) {
            if (right + 1 >= n || (left > 0 && nums[left - 1] > nums[right + 1])) {
                left--;
                min = Math.min(min, nums[left]);
            } else {
                right++;
                min = Math.min(min, nums[right]);
            }
            ans = Math.max(ans, (right - left + 1) * min);
        }
        return ans;
    }

    // 柱装图中最大的矩形
    public int largestRectangleArea(int[] heights) {
        int n = heights.length;
        int[] left = new int[n];
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            int x = heights[i];
            while (!stack.isEmpty() && x <= heights[stack.peek()]) {
                stack.pop();
            }
            left[i] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(i);
        }
        int[] right = new int[n];
        stack.clear();
        for (int i = n - 1; i >= 0; i--) {
            int x = heights[i];
            while (!stack.isEmpty() && x <= heights[stack.peek()]) {
                stack.pop();
            }
            right[i] = stack.isEmpty() ? n : stack.peek();
            stack.push(i);
        }
        int ans = 0;
        for (int i = 0; i < n; i++) {
            ans = Math.max(ans, heights[i] * (right[i] - left[i] - 1));
        }
        return ans;
    }

    // 零钱兑换二
    public int change(int amount, int[] coins) {
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for (int coin : coins) {
            for (int i = coin; i <= amount; i++) {
                dp[i] += dp[i - coin];
            }
        }
        return dp[amount];
    }

    // 二叉树的右视图
    List<Integer> res = new ArrayList<>();

    public List<Integer> rightSideView(TreeNode root) {
        dfs(root, 0);
        return res;
    }

    private void dfs(TreeNode root, int dep) {
        if (root == null) {
            return;
        }
        if (dep == res.size()) {
            res.add(root.val);
        }
        dfs(root.right, dep + 1);
        dfs(root.left, dep + 1);
    }

    // 统计将重叠区间合并的方案数
    public int countWays(int[][] ranges) {
        Arrays.sort(ranges, (a, b) -> a[0] - b[0]);
        int ans = 1;
        int maxR = -1;
        for (int[] p : ranges) {
            if (p[0] > maxR) {
                ans = ans * 2 % 1000000007;
            }
            maxR = Math.max(maxR, p[1]);
        }
        return ans;
    }

    // 访问完所有房间的第一天
    public int firstDayBeenInAllRooms(int[] nextVisit) {
        final long MOD = 1000000007;
        int n = nextVisit.length;
        long[] s = new long[n];
        for (int i = 0; i < n - 1; i++) {
            int j = nextVisit[i];
            s[i + 1] = (s[i] * 2 - s[j] + 2 + MOD) % MOD;
        }
        return (int) s[n - 1];
    }

    // 故障键盘
    public String finalString(String s) {
        Deque<Character> q = new ArrayDeque<>();
        boolean tail = true;
        for (char c : s.toCharArray()) {
            if (c == 'i') {
                tail = !tail;
            } else if (tail) {
                q.addLast(c);
            } else {
                q.addFirst(c);
            }
        }
        StringBuilder sb = new StringBuilder();
        for (char c : q) {
            sb.append(c);
        }
        if (!tail) {
            sb.reverse();
        }
        return sb.toString();
    }

    // 所有可能的二叉树
    private static final List<TreeNode>[] f = new ArrayList[11];

    static {
        Arrays.setAll(f, i -> new ArrayList<>());
        f[1].add(new TreeNode());
        for (int i = 2; i < f.length; i++) { // 计算 f[i]
            for (int j = 1; j < i; j++) { // 枚举左子树叶子数
                for (TreeNode left : f[j]) { // 枚举左子树
                    for (TreeNode right : f[i - j]) { // 枚举右子树
                        f[i].add(new TreeNode(0, left, right));
                    }
                }
            }
        }
    }

    public List<TreeNode> allPossibleFBT(int n) {
        return f[n % 2 > 0 ? (n + 1) / 2 : 0];
    }

    // 找出克隆二叉树中的相同节点
    public final TreeNode getTargetCopy(final TreeNode original, final TreeNode cloned, final TreeNode target) {
        if (original == null || original == target) {
            return cloned;
        }
        TreeNode left = getTargetCopy(original.left, cloned.left, target);
        if (left != null) {
            return left;
        }
        return getTargetCopy(original.right, cloned.right, target);
    }

    // 会议室
    public int minMeetingRooms(int[][] intervals) {
        if (intervals == null || intervals.length == 0) {
            return 0;
        }
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
        pq.add(intervals[0][1]);
        for (int i = 1; i < intervals.length; i++) {
            int last = pq.peek();
            if (last <= intervals[i][0]) {
                pq.poll();
                pq.add(intervals[i][1]);
            } else {
                pq.add(intervals[i][1]);
            }
        }
        return pq.size();
    }

    // 有向无环图中一个节点的所有祖先
    public List<List<Integer>> getAncestors(int n, int[][] edges) {
        List<Integer>[] g = new ArrayList[n];
        Arrays.setAll(g, i -> new ArrayList<>());
        for (int[] e : edges) {
            g[e[1]].add(e[0]);
        }
        List<Integer>[] ans = new ArrayList[n];
        Arrays.setAll(ans, i -> new ArrayList<>());
        boolean[] vis = new boolean[n];
        for (int i = 0; i < n; i++) {
            Arrays.fill(vis, false);
            dfs(i, g, vis);
            vis[i] = false;
            for (int j = 0; j < n; j++) {
                if (vis[j]) {
                    ans[i].add(j);
                }
            }
        }
        return Arrays.asList(ans);
    }

    private void dfs(int i, List<Integer>[] g, boolean[] vis) {
        vis[i] = true;
        for (int y : g[i]) {
            if (!vis[y]) {
                dfs(y, g, vis);
            }
        }
    }

    // 节点与其祖先之间的最大差值
    private int ans;

    public int maxAncestorDiff(TreeNode root) {
        dfs(root, root.val, root.val);
        return ans;
    }

    private void dfs(TreeNode root, int min, int max) {
        if (root == null) {
            ans = Math.max(ans, max - min);
            return;
        }
        min = Math.min(min, root.val);
        max = Math.max(max, root.val);
        dfs(root.left, min, max);
        dfs(root.right, min, max);
    }

    // 简化路径
    public String simplifyPath(String path) {
        Deque<String> stack = new LinkedList<>();
        String[] split = path.split("/");
        for (String s : split) {
            if (s.equals("..")) {
                if (!stack.isEmpty()) {
                    stack.pop();
                }
            } else if (!s.isEmpty() && !s.equals(".")) {
                stack.push(s);
            }
        }
        String res = "";
        for (String d : stack) {
            res = "/" + d + res;
        }
        return res.isEmpty() ? "/" : res;
    }

    // 分发饼干
    public int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        int m = g.length, n = s.length;
        int cnt = 0;
        for (int i = 0, j = 0; i < m && j < n; j++, i++) {
            while (j < n && g[i] > s[j]) {
                j++;
            }
            if (j < n) {
                cnt++;
            }
        }
        return cnt;
    }

    // 两整数之和
    public int getSum(int a, int b) {
        while (b != 0) {
            int carry = (a & b) << 1;
            a = a ^ b;
            b = carry;
        }
        return a;
    }

    // 删除有序数组中的重复项
    public int removeDuplicates(int[] nums) {
        int n = nums.length;
        int m = 1;
        for (int i = 1; i < n; i++) {
            if (nums[i] != nums[i - 1]) {
                nums[m++] = nums[i];
            }
        }
        return m;
    }

    // 使数组连续的最少操作数
    public int minOperations(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        int m = 1;
        for (int i = 1; i < n; i++) {
            if (nums[i] != nums[i - 1]) {
                nums[m++] = nums[i];
            }
        }
        int ans = 0;
        int left = 0;
        for (int i = 0; i < m; i++) {
            while (nums[left] < nums[i] - n + 1) {
                left++;
            }
            ans = Math.max(ans, i - left + 1);
        }
        return n - ans;
    }

    // 正整数和负整数的最大计数
    public int maximumCount(int[] nums) {
        int m = 0, n = 0;
        m = (int) Arrays.stream(nums).filter(x -> x < 0).count();
        n = (int) Arrays.stream(nums).filter(x -> x > 0).count();
        return Math.max(m, n);
    }

    // 字符串转换整数
    public int myAtoi(String s) {
        char[] c = s.trim().toCharArray();
        if (c.length == 0) {
            return 0;
        }
        int res = 0, board = Integer.MAX_VALUE / 10;
        int i = 1, sign = 1;
        if (c[0] == '-') {
            sign = -1;
        } else if (c[0] != '+') {
            i = 0;
        }
        for (int j = i; j < c.length; j++) {
            if (c[j] < '0' || c[j] > '9') {
                break;
            }
            if (res > board || res == board && c[j] > '7') {
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            }
            res = res * 10 + (c[j] - '0');
        }
        return sign * res;
    }

    // 修改后的最大二进制字符串
    public String maximumBinaryString(String binary) {
        int i = binary.indexOf('0');
        if (i < 0) {
            return binary;
        }
        char[] charArray = binary.toCharArray();
        int cnt = 0;
        for (i++; i < charArray.length; i++) {
            cnt += charArray[i] - '0';
        }
        return "1".repeat(charArray.length - 1 - cnt) + "0" + "1".repeat(cnt);
    }

    // Power(x,n)
    public double myPow(double x, int n) {
        if (x == 0.0f) {
            return 0.0d;
        }
        long b = n;
        double res = 1.0d;
        if (b < 0) {
            x = 1 / x;
            b = -b;
        }
        while (b > 0) {
            if ((b & 1) == 1) {
                res *= x;
            }
            x *= x;
            b >>= 1;
        }
        return res;
    }

    // 全排列二
    boolean[] vis;

    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        vis = new boolean[nums.length];
        Arrays.sort(nums);
        backtrace(nums, ans, 0, path);
        return ans;
    }

    private void backtrace(int[] nums, List<List<Integer>> ans, int idx, List<Integer> path) {
        if (idx == nums.length) {
            ans.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (vis[i] || (i > 0 && nums[i] == nums[i - 1] && !vis[i - 1])) {
                continue;
            }
            path.add(nums[i]);
            vis[i] = true;
            backtrace(nums, ans, idx + 1, path);
            vis[i] = false;
            path.remove(path.size() - 1);
        }
    }

    // 知道冠军一
    public int findChampion(int[][] grid) {
        int ans = 0;
        for (int i = 1; i < grid.length; i++) {
            if (grid[i][ans] == 1) {
                ans = i;
            }
        }
        return ans;
    }

    //
}

