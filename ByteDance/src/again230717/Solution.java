package again230717;

import java.util.*;

/**
 * 2023-07-17 another again
 */
public class Solution {


    public static void main(String[] args) {

    }

    // 字符串相加
    public static String addStrings(String num1, String num2) {
        StringBuilder sb = new StringBuilder();
        int m = num1.length() - 1, n = num2.length() - 1, carry = 0;
        while (m >= 0 || n >= 0 || carry > 0) {
            int sum = (m >= 0 ? num1.charAt(m) - '0' : 0) + (n >= 0 ? num2.charAt(n) - '0' : 0) + carry;
            carry = sum / 10;
            sum = sum % 10;
            sb.append(sum);
            m--;
            n--;
        }
        return sb.reverse().toString();
    }

    // 分割等和子集
    public boolean canPartition(int[] nums) {
        int n = nums.length;
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if ((sum & 1) == 1) {
            return false;
        }
        int target = sum / 2;
        boolean[] dp = new boolean[target + 1];
        dp[0] = true;
        for (int i = 0; i < n; i++) {
            int num = nums[i];
            for (int j = target; j >= num; j--) {
                dp[j] |= dp[j - num];
            }
        }
        return dp[target];
    }

    // 狒狒吃香蕉
    public int minEatingSpeed(int[] piles, int h) {
        int l = 1, r = Arrays.stream(piles).max().getAsInt();
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (getTime(piles, mid) > h) {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return l;
    }

    private int getTime(int[] piles, int k) {
        int cnt = 0;
        for (int p : piles) {
            cnt += p / k;
            if (p % k > 0) {
                cnt++;
            }
        }
        return cnt;
    }

    // 展平二叉搜索树
    public TreeNode increasingBST(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        inOrder(root, res);
        TreeNode dummy = new TreeNode(-1);
        TreeNode cur = dummy;
        for (int val : res) {
            cur.right = new TreeNode(val);
            cur = cur.right;
        }
        return dummy.right;
    }

    private void inOrder(TreeNode root, List<Integer> res) {
        if (root == null) {
            return;
        }
        inOrder(root.left, res);
        res.add(root.val);
        inOrder(root.right, res);
    }

    // 数组中和为0 的三个数
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int j = i + 1, k = n - 1;
            while (j < k) {
                while (j > i + 1 && j < n && nums[j] == nums[j - 1]) {
                    j++;
                }
                if (j >= k) {
                    break;
                }
                int sum = nums[i] + nums[j] + nums[k];
                if (sum == 0) {
                    ans.add(Arrays.asList(nums[i], nums[j], nums[k]));
                    j++;
                } else if (sum > 0) {
                    k--;
                } else {
                    j++;
                }
            }
        }
        return ans;
    }

    // 二叉树的右侧视图
    List<Integer> ans = new ArrayList<>();

    public List<Integer> rightSideView(TreeNode root) {
        dfs(root, 0);
        return ans;
    }

    private void dfs(TreeNode root, int depth) {
        if (root == null) {
            return;
        }
        if (depth == ans.size()) {
            ans.add(root.val);
        }
        dfs(root.right, depth + 1);
        dfs(root.left, depth + 1);
    }

    // 宝石与石头
    public int numJewelsInStones(String jewels, String stones) {
        int ans = 0;
        Set<Character> set = new HashSet<>();
        int m = jewels.length(), n = stones.length();
        for (int i = 0; i < m; i++) {
            char c = jewels.charAt(i);
            set.add(c);
        }
        for (int i = 0; i < n; i++) {
            char c = stones.charAt(i);
            if (set.contains(c)) {
                ans++;
            }
        }
        return ans;
    }

    // 每日温度
    public int[] dailyTemperatures(int[] temperatures) {
        Deque<Integer> deque = new ArrayDeque<>();
        int n = temperatures.length;
        int[] ans = new int[n];
        for (int i = 0; i < n; i++) {
            while (!deque.isEmpty() && temperatures[deque.peek()] < temperatures[i]) {
                int idx = deque.pop();
                ans[idx] = i - idx;
            }
            deque.push(i);
        }
        return ans;
    }

    // 将数组和减半的最少操作次数
    public int halveArray(int[] nums) {
        double s = 0;
        PriorityQueue<Double> pq = new PriorityQueue<>((a, b) -> Double.compare(b, a));
        for (int v : nums) {
            pq.offer(v * 1.0);
            s += v;
        }
        s /= 2.0;
        int ans = 0;
        while (s > 0) {
            double t = pq.poll();
            s -= t / 2.0;
            pq.offer(t / 2.0);
            ++ans;
        }
        return ans;
    }

    // 解决智力问题
    public long mostPoints(int[][] questions) {
        var n = questions.length;
        long[] f = new long[n + 1];
        for (int i = n - 1; i >= 0; i--) {
            int[] q = questions[i];
            int j = i + q[1] + 1;
            f[i] = Math.max(f[i + 1], q[0] + (j < n ? f[j] : 0));
        }
        return f[0];
    }

    // 环形链表
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode slow = head;
        ListNode fast = head.next;
        while (slow != fast) {
            if (fast.next == null || fast.next.next == null) {
                return false;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        return true;
    }

    // 环形链表二
    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null || head.next.next == null) {
            return null;
        }
        ListNode slow = head.next;
        ListNode fast = head.next.next;
        while (slow != fast) {
            if (fast.next == null || fast.next.next == null) {
                return null;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        fast = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }

    // 爬楼梯
    public int climbStairs(int n) {
        if (n <= 2) {
            return n;
        }
        int[] f = new int[n + 1];
        f[1] = 1;
        f[2] = 2;
        for (int i = 3; i <= n; i++) {
            f[i] = f[i - 1] + f[i - 2];
        }
        return f[n];
    }

    // 零钱兑换
    int INF = Integer.MAX_VALUE;

    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        // 除以二是为了防止下面+1 后溢出
        Arrays.fill(dp, INF / 2);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (coin <= i) {
                    dp[i] = Math.min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        return dp[amount] == INF / 2 ? -1 : dp[amount];
    }

    // 重排链表
    public void reorderList(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode cur = slow.next;
        slow.next = null;
        // 反转 cur
        ListNode pre = null;
        while (cur != null) {
            ListNode t = cur.next;
            cur.next = pre;
            pre = cur;
            cur = t;
        }
        cur = head;
        //合并
        while (pre != null) {
            ListNode t = pre.next;
            pre.next = cur.next;
            cur.next = pre;
            cur = pre.next;
            pre = t;
        }
    }

    // H 指数
    public int hIndex(int[] citations) {
        Arrays.sort(citations);
        int n = citations.length;
        for (int i = 0; i < n; i++) {
            int remain = n - i;
            if (remain <= citations[i]) {
                return remain;
            }
        }
        return 0;
    }

    // 翻转卡片游戏
    public int flipgame(int[] fronts, int[] backs) {
        Set<Integer> forbidden = new HashSet<>();
        for (int i = 0; i < fronts.length; i++) {
            if (fronts[i] == backs[i]) {
                forbidden.add(fronts[i]);
            }
        }
        int ans = Integer.MAX_VALUE;
        for (int x : fronts) {
            if (!forbidden.contains(x)) {
                ans = Math.min(ans, x);
            }
        }
        for (int x : backs) {
            if (!forbidden.contains(x)) {
                ans = Math.min(ans, x);
            }
        }
        return ans == Integer.MAX_VALUE ? 0 : ans;
    }

    // 乘积小于 k 的子数组
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        int n = nums.length, ans = 0;
        if (k < 1) {
            return 0;
        }
        for (int i = 0, j = 0, cur = 1; i < n; i++) {
            cur *= nums[i];
            while (cur >= k) {
                cur /= nums[j++];
            }
            ans += i - j + 1;
        }
        return ans;
    }

    // 和为 k 的子数组
    public int subarraySum(int[] nums, int k) {
        int n = nums.length, ans = 0;
        int[] sum = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            sum[i] = sum[i - 1] + nums[i - 1];
        }
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        for (int i = 1; i <= n; i++) {
            int t = sum[i], d = t - k;
            ans += map.getOrDefault(d, 0);
            map.put(t, map.getOrDefault(t, 0) + 1);
        }
        return ans;
    }

    // 0 和 1个数相同的子数组
    public int findMaxLength(int[] nums) {
        int n = nums.length, ans = 0;
        int[] sum = new int[n + 10];
        for (int i = 1; i <= n; i++) {
            sum[i] = sum[i - 1] + (nums[i - 1] == 0 ? -1 : 1);
        }
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 0);
        for (int i = 1; i <= n; i++) {
            int t = sum[i];
            if (map.containsKey(t)) {
                ans = Math.max(ans, i - map.get(t));
            }
            if (!map.containsKey(t)) {
                map.put(t, i);
            }
        }
        return ans;
    }
}
