package start230901;

import java.util.*;

public class Solution {

    // 买铅笔和钢笔的方案数
    public long waysToBuyPensPencils(int total, int cost1, int cost2) {
        if (cost1 < cost2) {
            return waysToBuyPensPencils(total, cost2, cost1);
        }
        long res = 0, choose = 0;
        while (choose * cost1 <= total) {
            res += (total - choose * cost1) / cost2 + 1;
            choose++;
        }
        return res;
    }

    // 消灭怪物的最大数量
    public int eliminateMaximum(int[] dist, int[] speed) {
        int n = dist.length;
        int[] arrivalTimes = new int[n];
        for (int i = 0; i < n; i++) {
            arrivalTimes[i] = (dist[i] - 1) / speed[i] + 1;
        }
        Arrays.sort(arrivalTimes);
        for (int i = 0; i < n; i++) {
            if (arrivalTimes[i] <= i) {
                return i;
            }
        }
        return n;
    }

    // 从两个数字数组里生成最小数字
    public int minNumber(int[] nums1, int[] nums2) {
        int ans = 100;
        for (int a : nums1) {
            for (int b : nums2) {
                if (a == b) {
                    ans = Math.min(ans, a);
                } else {
                    ans = Math.min(ans, Math.min(a * 10 + b, b * 10 + a));
                }
            }
        }
        return ans;
    }

    // 最多可以摧毁的敌人堡垒数目
    public int captureForts(int[] forts) {
        int n = forts.length;
        int ans = 0, pre = -1;
        for (int i = 0; i < n; i++) {
            if (forts[i] == 1 || forts[i] == -1) {
                if (pre >= 0 && forts[i] != forts[pre]) {
                    ans = Math.max(ans, i - pre - 1);
                }
                pre = i;
            }
        }
        return ans;
    }

    // 最深叶节点的最近公共祖先
    private TreeNode treeNodeAns;
    private int maxDepth = -1;

    public TreeNode lcaDeepestLeaves(TreeNode root) {
        dfs(root, 0);
        return treeNodeAns;
    }

    private int dfs(TreeNode root, int depth) {
        if (root == null) {
            maxDepth = Math.max(maxDepth, depth);
            return depth;
        }
        int leftDepth = dfs(root.left, depth + 1);
        int rightDepth = dfs(root.right, depth + 1);
        if (leftDepth == rightDepth && leftDepth == maxDepth) {
            treeNodeAns = root;
        }
        return Math.max(leftDepth, rightDepth);
    }

    // 修车的最少时间
    public long repairCars(int[] ranks, int cars) {
        long l = 0, r = 1L * ranks[0] * cars * cars;
        while (l < r) {
            long m = l + r >> 1;
            if (check(ranks, cars, m)) {
                r = m;
            } else {
                l = m + 1;
            }
        }
        return l;
    }

    private boolean check(int[] ranks, int cars, long m) {
        long cnt = 0;
        for (int x : ranks) {
            cnt += (long) Math.sqrt(m / x);
        }
        return cnt >= cars;
    }

    // 每个小孩最多能分到多少糖果
    public int maximumCandies(int[] candies, long k) {
        int max = Arrays.stream(candies).max().getAsInt();
        int left = 0, right = max;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (getKids(candies, mid) < k) {
                right = mid - 1;
            } else {
                left = mid;
            }
        }
        return left;
    }

    private long getKids(int[] candies, int mid) {
        long cnt = 0;
        for (int candy : candies) {
            cnt += candy / mid;
        }
        return cnt;
    }

    // 计算列车到站时间
    public int findDelayedArrivalTime(int arrivalTime, int delayedTime) {
        return (arrivalTime + delayedTime) % 24;
    }

    // 最长回文串
    public int longestPalindrome(String s) {
        int[] arr = new int[128];
        for (char c : s.toCharArray()) {
            arr[c]++;
        }
        int count = 0;
        for (int i : arr) {
            count += (i % 2);
        }
        return count == 0 ? s.length() : (s.length() - count + 1);
    }

    // 多个数组求交集
    public List<Integer> intersection(int[][] nums) {
        int[] cnt = new int[10001];
        int n = nums.length;
        for (int[] num : nums) {
            for (int x : num) {
                cnt[x]++;
            }
        }
        List<Integer> ans = new ArrayList<>();
        for (int i = 0; i <= 1000; i++) {
            if (cnt[i] == n) {
                ans.add(i);
            }
        }
        return ans;
    }

    // 课程表
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        if (numCourses <= 0) {
            return new int[0];
        }
        HashSet<Integer>[] adj = new HashSet[numCourses];
        for (int i = 0; i < numCourses; i++) {
            adj[i] = new HashSet<>();
        }
        int[] inDegree = new int[numCourses];
        for (int[] p : prerequisites) {
            adj[p[1]].add(p[0]);
            inDegree[p[0]]++;
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (inDegree[i] == 0) {
                queue.offer(i);
            }
        }
        int[] res = new int[numCourses];
        int count = 0;
        while (!queue.isEmpty()) {
            Integer poll = queue.poll();
            res[count++] = poll;
            Set<Integer> tmp = adj[poll];
            for (Integer num : tmp) {
                inDegree[num]--;
                if (inDegree[num] == 0) {
                    queue.offer(num);
                }
            }
        }
        if (count == numCourses) {
            return res;
        }
        return new int[0];
    }

    // 最大子数组和
    public int maxSubArray(int[] nums) {
        int ans = nums[0], pre = nums[0];
        for (int i = 1; i < nums.length; i++) {
            pre = Math.max(pre + nums[i], nums[i]);
            ans = Math.max(ans, pre);
        }
        return ans;
    }

    // 删除链表的倒数第 N 个节点
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0, head);
        ListNode first = head;
        ListNode second = dummy;
        for (int i = 0; i < n; i++) {
            first = first.next;
        }
        while (first != null) {
            first = first.next;
            second = second.next;
        }
        second.next = second.next.next;
        return dummy.next;
    }

    // 课程表四
    public List<Boolean> checkIfPrerequisite(int numCourses, int[][] prerequisites, int[][] queries) {
        boolean[][] f = new boolean[numCourses][numCourses];
        List<Integer>[] g = new List[numCourses];
        int[] indeg = new int[numCourses];
        Arrays.setAll(g, i -> new ArrayList<>());
        for (int[] p : prerequisites) {
            g[p[0]].add(p[1]);
            ++indeg[p[1]];
        }
        Deque<Integer> deque = new ArrayDeque<>();
        for (int i = 0; i < numCourses; i++) {
            if (indeg[i] == 0) {
                deque.offer(i);
            }
        }
        while (!deque.isEmpty()) {
            Integer i = deque.poll();
            for (int j : g[i]) {
                f[i][j] = true;
                for (int h = 0; h < numCourses; h++) {
                    f[h][j] |= f[h][i];
                }
                if (--indeg[j] == 0) {
                    deque.offer(j);
                }
            }
        }
        List<Boolean> ans = new ArrayList<>();
        for (int[] q : queries) {
            ans.add(f[q[0]][q[1]]);
        }
        return ans;
    }

    // 子集二
    List<List<Integer>> ans;
    List<Integer> path;

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        ans = new ArrayList<>();
        path = new ArrayList<>();
        Arrays.sort(nums);
        int n = nums.length;
        boolean[] vis = new boolean[n];
        backtrace(nums, 0, vis, 0);
        return ans;
    }

    private void backtrace(int[] nums, int start, boolean[] vis, int n) {
        ans.add(new ArrayList<>(path));
        for (int i = start; i < n; i++) {
            if (i > 0 && nums[i - 1] == nums[i] && !vis[i - 1]) {
                continue;
            }
            vis[i] = true;
            path.add(nums[i]);
            backtrace(nums, i + 1, vis, n);
            vis[i] = false;
            path.remove(path.size() - 1);
        }
    }

    // 可以攻击国王的皇后
    private final static int[][] directions = {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};

    public List<List<Integer>> queensAttacktheKing(int[][] queens, int[] king) {
        boolean[][] check = new boolean[8][8];
        for (int[] q : queens) {
            check[q[0]][q[1]] = true;
        }
        List<List<Integer>> ans = new ArrayList<>();
        for (int[] d : directions) {
            int x = king[0] + d[0];
            int y = king[1] + d[1];
            while (0 <= x && x < 8 && 0 <= y && y < 8) {
                if (check[x][y]) {
                    ans.add(List.of(x, y));
                    break;
                }
                x += d[0];
                y += d[1];
            }
        }
        return ans;
    }

    // 划分字母区间
    public List<Integer> partitionLabels(String s) {
        List<Integer> ans = new ArrayList<>();
        int n = s.length();
        int[] maxIndex = new int[26];
        for (int i = 0; i < n; i++) {
            maxIndex[s.charAt(i) - 'a'] = i;
        }
        int start = 0, end = 0;
        for (int i = 0; i < n; i++) {
            end = Math.max(end, maxIndex[s.charAt(i) - 'a']);
            if (i == end) {
                ans.add(end - start + 1);
                start = end + 1;
            }
        }
        return ans;
    }

    // 反转链表
    public ListNode reverseList(ListNode head) {
        ListNode cur = head, pre = null;
        while (cur != null) {
            ListNode tmp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = tmp;
        }
        return pre;
    }

    // 宝石补给
    public int giveGem(int[] gem, int[][] operations) {
        for (int[] op : operations) {
            int tmp = gem[op[0]] / 2;
            gem[op[0]] -= tmp;
            gem[op[1]] += tmp;
        }
        return Arrays.stream(gem).max().getAsInt() - Arrays.stream(gem).min().getAsInt();
    }

    // 单词的压缩编码
    Trie trie = new Trie();

    public int minimumLengthEncoding(String[] words) {
        for (String word : words) {
            trie.insert(word);
        }
        int res = 0;
        Set<String> set = new HashSet<>();
        for (String word : words) {
            if (set.contains(word)) {
                continue;
            }
            set.add(word);
            boolean maxWord = trie.search(word);
            if (maxWord) {
                res = res + word.length() + 1;
            }
        }
        return res;
    }

    class Trie {
        TrieNode root = new TrieNode();

        public void insert(String word) {
            TrieNode cur = root;
            boolean newBranch = false;
            for (int i = word.length() - 1; i >= 0; i--) {
                char c = word.charAt(i);
                if (cur.next[c - 'a'] == null) {
                    cur.next[c - 'a'] = new TrieNode();
                    newBranch = true;
                }
                cur = cur.next[c - 'a'];
                if (cur.isEnd && i != 0) {
                    cur.isEnd = false;
                }
            }
            if (newBranch) {
                cur.isEnd = true;
            }
        }

        public boolean search(String word) {
            TrieNode cur = root;
            for (int i = word.length() - 1; i >= 0; i--) {
                char c = word.charAt(i);
                if (cur.next[c - 'a'] == null) {
                    return false;
                }
                cur = cur.next[c - 'a'];
            }
            return cur.isEnd;
        }
    }

    class TrieNode {
        TrieNode[] next = new TrieNode[26];
        boolean isEnd;
    }

    // 打家劫舍
    public int rob(int[] nums) {
        int pre = 0, cur = 0;
        for (int num : nums) {
            int tmp = Math.max(cur, pre + num);
            pre = cur;
            cur = tmp;
        }
        return cur;
    }

    // 打家劫舍二
    public int robG2(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }
        return Math.max(myRob(Arrays.copyOfRange(nums, 0, nums.length - 1)), myRob(Arrays.copyOfRange(nums, 1, nums.length)));
    }

    public int myRob(int[] nums) {
        int pre = 0, cur = 0;
        for (int num : nums) {
            int tmp = Math.max(pre + num, cur);
            pre = cur;
            cur = tmp;
        }
        return cur;
    }

    // 七进制数
    public String convertToBase7(int num) {
        boolean flag = num < 0;
        if (flag) {
            num = -num;
        }
        StringBuffer sb = new StringBuffer();
        do {
            sb.append(num % 7);
            num /= 7;
        } while (num != 0);
        sb.reverse();
        return flag ? "-" + sb.toString() : sb.toString();
    }

    // 拿硬币
    public int minCount(int[] coins) {
        int ans = 0;
        for (int num : coins) {
            ans += (num + 1) / 2;
        }
        return ans;
    }

    // 二叉数的最大深度
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = maxDepth(root.left);
        int right = maxDepth(root.right);
        return Math.max(left, right) + 1;
    }

    // 将钱分给最多的儿童
    public int distMoney(int money, int children) {
        if (money < children) {
            return -1;
        }
        if (money > 8 * children) {
            return children - 1;
        }
        if (money == 8 * children - 4) {
            return children - 2;
        }
        return (money - children) / 7;
    }

    // 递枕头
    public int passThePillow(int n, int time) {
        int k = time / (n - 1);
        int mod = time % (n - 1);
        return (k & 1) == 1 ? n - mod : mod + 1;
    }

    // 餐厅过滤器
    public List<Integer> filterRestaurants(int[][] restaurants, int veganFriendly, int maxPrice, int maxDistance) {
        Arrays.sort(restaurants, (a, b) -> a[1] == b[1] ? b[0] - a[0] : b[1] - a[1]);
        List<Integer> ans = new ArrayList<>();
        for (int[] r : restaurants) {
            if (r[2] >= veganFriendly && r[3] <= maxPrice && r[4] <= maxDistance) {
                ans.add(r[0]);
            }
        }
        return ans;
    }
}


