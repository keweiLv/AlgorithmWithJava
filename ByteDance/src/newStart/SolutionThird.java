package newStart;

import java.util.*;

public class SolutionThird {

    // 可被三整除的偶数的平均值
    public int averageValue(int[] nums) {
        int s = 0, n = 0;
        for (int num : nums) {
            if (num % 3 == 0 && num % 2 == 0) {
                s += num;
                n++;
            }
        }
        return n == 0 ? 0 : s / n;
    }

    // 二叉树最大宽度
    Map<Integer, Integer> map = new HashMap<>();
    int ans;

    public int widthOfBinaryTree(TreeNode root) {
        dfs(root, 1, 0);
        return ans;
    }

    private void dfs(TreeNode root, int v, int depth) {
        if (root == null) {
            return;
        }
        if (!map.containsKey(depth)) {
            map.put(depth, v);
        }
        ans = Math.max(ans, v - map.get(depth) + 1);
        v = v - map.get(depth) + 1;
        dfs(root.left, v << 1, depth + 1);
        dfs(root.right, v << 1 | 1, depth + 1);
    }

    // 删点成林
    public List<TreeNode> delNodes(TreeNode root, int[] to_delete) {
        List<TreeNode> ans = new ArrayList<>();
        Set<Integer> set = new HashSet<>();
        for (int i : to_delete) {
            set.add(i);
        }
        if (dfs(root, ans, set) != null) {
            ans.add(root);
        }
        return ans;
    }

    private TreeNode dfs(TreeNode root, List<TreeNode> ans, Set<Integer> set) {
        if (root == null) {
            return null;
        }
        root.left = dfs(root.left, ans, set);
        root.right = dfs(root.right, ans, set);
        if (!set.contains(root.val)) {
            return root;
        }
        if (root.left != null) {
            ans.add(root.left);
        }
        if (root.right != null) {
            ans.add(root.right);
        }
        return null;
    }

    // 数组中两个数的最大异或值
    public int findMaximumXOR(int[] nums) {
        int res = 0;
        int mask = 0;
        for (int i = 30; i >= 0; i--) {
            mask = mask | (1 << i);
            Set<Integer> set = new HashSet<>();
            for (int num : nums) {
                set.add(num & mask);
            }
            int temp = res | (1 << i);
            for (Integer pre : set) {
                if (set.contains(temp ^ pre)) {
                    res = temp;
                    break;
                }
            }
        }
        return res;
    }

    // 使用最小花费爬楼梯
    public int minCostClimbingStairs(int[] cost) {
        int n = cost.length;
        int[] dp = new int[n + 1];
        dp[0] = dp[1] = 0;
        for (int i = 2; i <= n; i++) {
            dp[i] = Math.min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2]);
        }
        return dp[n];
    }

    // 叶值的最小代价生成树
    public int mctFromLeafValues(int[] arr) {
        Deque<Integer> stack = new ArrayDeque<>();
        stack.offerLast(Integer.MAX_VALUE);
        int ans = 0;
        for (int i = 0; i < arr.length; i++) {
            while (arr[i] > stack.peekLast()) {
                ans += stack.pollLast() * Math.min(stack.peekLast(), arr[i]);
            }
            stack.offerLast(arr[i]);
        }
        while (stack.size() > 2) {
            ans += stack.pollLast() * stack.peekLast();
        }
        return ans;
    }

    // 礼盒的最大甜蜜度
    public int maximumTastiness(int[] price, int k) {
        Arrays.sort(price);
        int left = 0, right = (price[price.length - 1] - price[0]) / (k - 1) + 1;
        while (left + 1 < right) {
            int mid = left + (right - left) / 2;
            if (check(price, mid) >= k) {
                left = mid;
            } else {
                right = mid;
            }
        }
        return left;
    }

    private int check(int[] price, int d) {
        int cnt = 1, pre = price[0];
        for (int p : price) {
            if (p >= pre + d) {
                cnt++;
                pre = p;
            }
        }
        return cnt;
    }

    // 爱吃香蕉的珂珂
    public int minEatingSpeed(int[] piles, int h) {
        int l = 0, r = (int) 1e9;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (check(piles, mid, h)) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }

    private boolean check(int[] piles, int mid, int h) {
        int ans = 0;
        for (int p : piles) {
            ans += Math.ceil(p * 1.0 / mid);
        }
        return ans <= h;
    }

    // 统计范围内的元音字符串数
    public int[] vowelStrings(String[] words, int[][] queries) {
        Set<Character> set = Set.of('a', 'e', 'i', 'o', 'u');
        int n = words.length;
        int[] s = new int[n + 1];
        for (int i = 0; i < n; i++) {
            char a = words[i].charAt(0), b = words[i].charAt(words[i].length() - 1);
            s[i + 1] = s[i] + (set.contains(a) && set.contains(b) ? 1 : 0);
        }
        int m = queries.length;
        int[] ans = new int[m];
        for (int i = 0; i < m; i++) {
            int l = queries[i][0], r = queries[i][1];
            ans[i] = s[r + 1] - s[l];
        }
        return ans;
    }

    // 删除并获得点数
    int[] cnts = new int[10010];

    public int deleteAndEarn(int[] nums) {
        int n = nums.length;
        int max = 0;
        for (int nu : nums) {
            cnts[nu]++;
            max = Math.max(max, nu);
        }
        int[][] f = new int[max + 1][2];
        for (int i = 1; i <= max; i++) {
            f[i][0] = Math.max(f[i - 1][0], f[i - 1][1]);
            f[i][1] = f[i - 1][0] + i * cnts[i];
        }
        return Math.max(f[max][0], f[max][1]);
    }

    // 最长湍流子数组
    public int maxTurbulenceSize(int[] arr) {
        int n = arr.length, ans = 1;
        int[][] f = new int[n][2];
        f[0][0] = f[0][1] = 1;
        for (int i = 1; i < n; i++) {
            f[i][0] = f[i][1] = 1;
            if (arr[i] > arr[i - 1]) {
                f[i][0] = f[i - 1][1] + 1;
            } else if (arr[i] < arr[i - 1]) {
                f[i][1] = f[i - 1][0] + 1;
            }
            ans = Math.max(ans, Math.max(f[i][0], f[i][1]));
        }
        return ans;
    }

    // 单字符重复子串的最大长度
    public int maxRepOpt1(String text) {
        int[] cnt = new int[26];
        int n = text.length();
        for (int i = 0; i < n; i++) {
            cnt[text.charAt(i) - 'a']++;
        }
        int ans = 0, i = 0;
        while (i < n) {
            int j = i;
            while (j < n && text.charAt(j) == text.charAt(i)) {
                j++;
            }
            int l = j - i;
            int k = j + 1;
            while (k < n && text.charAt(k) == text.charAt(i)) {
                k++;
            }
            int r = k - j - 1;
            ans = Math.max(ans, Math.min(l + r + 1, cnt[text.charAt(i) - 'a']));
        }
        return ans;
    }

    // 最长公共子序列
    public int longestCommonSubsequence(String text1, String text2) {
        int n = text1.length(), m = text2.length();
        text1 = " " + text1;
        text2 = " " + text2;
        char[] cs1 = text1.toCharArray(), cs2 = text2.toCharArray();
        int[][] f = new int[n + 1][m + 1];
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (cs1[i] == cs2[j]) {
                    f[i][j] = f[i - 1][j - 1] + 1;
                } else {
                    f[i][j] = Math.max(f[i - 1][j], f[i][j - 1]);
                }
            }
        }
        return f[n][m];
    }

    // 不同的平均值数目
    public int distinctAverages(int[] nums) {
        Arrays.sort(nums);
        Set<Integer> set = new HashSet<>();
        int n = nums.length;
        for (int i = 0; i < n >> 1; i++) {
            set.add(nums[i] + nums[n - i - 1]);
        }
        return set.size();
    }

    // 两个字符串的删除操作
    public int minDistance(String word1, String word2) {
        char[] cs1 = word1.toCharArray();
        char[] cs2 = word2.toCharArray();
        int n = word1.length(), m = word2.length();
        int[][] f = new int[n + 1][m + 1];
        for (int i = 0; i <= n; i++) {
            f[i][0] = 1;
        }
        for (int j = 0; j <= m; j++) {
            f[0][j] = 1;
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                f[i][j] = Math.max(f[i - 1][j], f[i][j - 1]);
                if (cs1[i - 1] == cs2[j - 1]) {
                    f[i][j] = Math.max(f[i][j], f[i - 1][j - 1] + 1);
                }
            }
        }
        int max = f[n][m] - 1;
        return n - max + m - max;
    }

    // 对数组执行操作
    public int[] applyOperations(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n - 1; i++) {
            if (nums[i] == nums[i + 1]) {
                nums[i] <<= 1;
                nums[i + 1] = 0;
            }
        }
        int[] ans = new int[n];
        int i = 0;
        for (int x : nums) {
            if (x > 0) {
                ans[i++] = x;
            }
        }
        return ans;
    }

    // 相等行列对
    public int equalPairs(int[][] grid) {
        int n = grid.length;
        int ans = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int ok = 1;
                for (int k = 0; k < n; k++) {
                    if (grid[i][k] != grid[k][j]) {
                        ok = 0;
                        break;
                    }
                }
                ans += ok;
            }
        }
        return ans;
    }

    // 翻转链表
    public ListNode reverseList(ListNode head) {
        ListNode pre = null, cur = head;
        while (cur != null) {
            ListNode tmp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = tmp;
        }
        return pre;
    }

    // 链表中倒数第K个节点
    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode fast = head, slow = head;
        for (int i = 0; i < k; i++) {
            fast = fast.next;
        }
        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }

    // 复杂链表的复制
    class Node {
        int val;
        Node next;
        Node random;

        public Node(int val) {
            this.val = val;
            this.next = null;
            this.random = null;
        }
    }

    public Node copyRandomList(Node head) {
        if (head == null) {
            return head;
        }
        Node cur = head;
        Map<Node, Node> map = new HashMap<>();
        while (cur != null) {
            map.put(cur, new Node(cur.val));
            cur = cur.next;
        }
        cur = head;
        while (cur != null) {
            map.get(cur).next = map.get(cur.next);
            map.get(cur).random = map.get(cur.random);
            cur = cur.next;
        }
        return map.get(head);
    }

    // 两个链表的第一个公共节点
    ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode A = headA, B = headB;
        while (A != B) {
            A = A != null ? A.next : headA;
            B = B != null ? B.next : headA;
        }
        return A;
    }

    // 从上到下打印二叉数二
    public List<List<Integer>> levelOrderTwo(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<List<Integer>> res = new ArrayList<>();
        Deque<TreeNode> stack = new LinkedList<>();
        stack.offerLast(root);
        while (!stack.isEmpty()) {
            List<Integer> list = new ArrayList<>();
            int n = stack.size();
            for (int i = 0; i < n; i++) {
                TreeNode node = stack.pollFirst();
                list.add(node.val);
                if (node.left != null) {
                    stack.offerLast(node.left);
                }
                if (node.right != null) {
                    stack.offerLast(node.right);
                }
            }
            res.add(list);
        }
        return res;
    }

    // 从上到下打印二叉数三
    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<List<Integer>> res = new ArrayList<>();
        Deque<TreeNode> stack = new LinkedList<>();
        stack.add(root);
        int n = stack.size();
        while (!stack.isEmpty()) {
            List<Integer> list = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                TreeNode node = stack.poll();
                list.add(node.val);
                if (node.left != null) {
                    stack.add(node.left);
                }
                if (node.right != null) {
                    stack.add(node.right);
                }
            }
            if (res.size() % 2 == 1) {
                Collections.reverse(list);
            }
            res.add(list);
        }
        return res;
    }

    // 有界数组中指定下标处的最大值
    public int maxValue(int n, int index, int maxSum) {
        int left = 1, right = maxSum;
        while (left < right) {
            int mid = (left + right + 1) / 2;
            if (sum(mid - 1, index) + sum(mid, n - index) <= maxSum) {
                left = mid;
            } else {
                right = mid - 1;
            }
        }
        return left;
    }

    private long sum(long x, int cnt) {
        return x >= cnt ? (x + x - cnt + 1) * cnt / 2 : (x + 1) * x / 2 + cnt - x;
    }

    // 路径总和
    int pathAns, t;

    public int pathSum(TreeNode root, int targetSum) {
        t = targetSum;
        dfs1(root);
        return pathAns;
    }

    private void dfs1(TreeNode root) {
        if (root == null) {
            return;
        }
        dfs2(root, root.val);
        dfs1(root.left);
        dfs1(root.right);
    }

    private void dfs2(TreeNode root, long val) {
        if (val == t) {
            pathAns++;
        }
        if (root.left != null) {
            dfs2(root.left, val + root.left.val);
        }
        if (root.right != null) {
            dfs2(root.right, val + root.right.val);
        }
    }

    // 目标和
    public int findTargetSumWays(int[] nums, int target) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        int diff = sum - target;
        if (diff < 0 || diff % 2 == 1) {
            return 0;
        }
        int n = nums.length, neg = diff / 2;
        int[][] dp = new int[n + 1][neg + 1];
        dp[0][0] = 1;
        for (int i = 1; i <= n; i++) {
            int num = nums[i - 1];
            for (int j = 0; j <= neg; j++) {
                dp[i][j] = dp[i - 1][j];
                if (j >= num) {
                    dp[i][j] += dp[i - 1][j - num];
                }
            }
        }
        return dp[n][neg];
    }

    // 比较字符串最小字母出现频次
    public int[] numSmallerByFrequency(String[] queries, String[] words) {
        int n = words.length;
        int[] nums = new int[n];
        for (int i = 0; i < n; i++) {
            nums[i] = f(words[i]);
        }
        Arrays.sort(nums);
        int m = queries.length;
        int[] ans = new int[m];
        for (int i = 0; i < m; i++) {
            int x = f(queries[i]);
            int l = 0, r = n;
            while (l < r) {
                int mid = (l + r) >> 1;
                if (nums[mid] > x) {
                    r = mid;
                } else {
                    l = mid + 1;
                }
            }
            ans[i] = n - l;
        }
        return ans;
    }

    private int f(String s) {
        int[] cnt = new int[26];
        for (int i = 0; i < s.length(); i++) {
            cnt[s.charAt(i) - 'a']++;
        }
        for (int x : cnt) {
            if (x > 0) {
                return x;
            }
        }
        return 0;
    }

    // 数组中第K个最大元素
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int nu : nums) {
            pq.add(nu);
            if (pq.size() > k) {
                pq.poll();
            }
        }
        return pq.peek();
    }
}
