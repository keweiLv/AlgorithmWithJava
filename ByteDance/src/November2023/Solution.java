package November2023;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

public class Solution {

    // 水果成篮
    public int totalFruit(int[] fruits) {
        int n = fruits.length, ans = 0;
        int[] cnts = new int[n + 10];
        for (int i = 0, j = 0, tol = 0; i < n; i++) {
            if (++cnts[fruits[i]] == 1) {
                tol++;
            }
            while (tol > 2) {
                if (--cnts[fruits[j++]] == 0) {
                    tol--;
                }
            }
            ans = Math.max(ans, i - j + 1);
        }
        return ans;
    }

    // K个不同整数的子数组
    public int subarraysWithKDistinct(int[] nums, int k) {
        int len = nums.length;
        int ans = 0;
        int count = 0;
        int left = 0;
        int right = 0;
        int[] map = new int[len + 1];
        while (right < len) {
            if (map[nums[right]] == 0) {
                count++;
            }
            map[nums[right]]++;
            right++;
            while (count > k) {
                map[nums[left]]--;
                if (map[nums[left]] == 0) {
                    count--;
                }
                left++;
            }
            ans += right - left;
        }
        return ans;
    }

    // 填充每个节点的下一个右侧节点指针二
    class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;

        public Node() {
        }

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, Node _left, Node _right, Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    }

    ;

    public Node connect(Node root) {
        Node res = root;
        if (root == null) {
            return res;
        }
        Deque<Node> deque = new ArrayDeque<>();
        deque.addLast(root);
        while (!deque.isEmpty()) {
            int sz = deque.size();
            while (sz-- > 0) {
                Node cur = deque.pollFirst();
                if (sz != 0) {
                    cur.next = deque.peekFirst();
                }
                if (cur.left != null) {
                    deque.addLast(cur.left);
                }
                if (cur.right != null) {
                    deque.addLast(cur.right);
                }
            }
        }
        return res;
    }

    // 分隔链表
    public ListNode[] splitListToParts(ListNode head, int k) {
        int cnt = 0;
        ListNode temp = head;
        while (temp != null && ++cnt > 0) {
            temp = temp.next;
        }
        int pre = cnt / k;
        ListNode[] ans = new ListNode[pre];
        for (int i = 0, sum = 1; i < k; i++, sum++) {
            ans[i] = head;
            temp = ans[i];
            int u = pre;
            while (u-- > 0 && ++sum > 0) {
                temp = temp.next;
            }
            int remain = k - i - 1;
            if (pre != 0 && sum + pre * remain < cnt && ++sum > 0) {
                temp = temp.next;
            }
            head = temp != null ? temp.next : null;
            if (temp != null) {
                temp.next = null;
            }
        }
        return ans;
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
            for (Integer prefix : set) {
                if (set.contains(prefix ^ temp)) {
                    res = temp;
                    break;
                }
            }
        }
        return res;
    }

    // 重复的DNA序列
    public List<String> findRepeatedDnaSequences(String s) {
        List<String> res = new ArrayList<>();
        int n = s.length();
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i + 10 <= n; i++) {
            String cur = s.substring(i, i + 10);
            int cnt = map.getOrDefault(cur, 0);
            if (cnt == 1) {
                res.add(cur);
            }
            map.put(cur, cnt + 1);
        }
        return res;
    }

    // 最大单词长度乘积
    public int maxProduct(String[] words) {
        Map<Integer, Integer> map = new HashMap<>();
        for (String word : words) {
            int t = 0, n = word.length();
            for (int i = 0; i < n; i++) {
                int c = word.charAt(i) - 'a';
                t |= (1 << c);
            }
            if (!map.containsKey(t) || map.get(t) < n) {
                map.put(t, n);
            }
        }
        int ans = 0;
        for (int a : map.keySet()) {
            for (int b : map.keySet()) {
                if ((a & b) == 0) {
                    ans = Math.max(ans, map.get(a) * map.get(b));
                }
            }
        }
        return ans;
    }

    // 优势洗牌
    public int[] advantageCount(int[] nums1, int[] nums2) {
        int n = nums1.length;
        int[] ans = new int[n];
        Arrays.sort(nums1);
        Integer[] array = IntStream.range(0, n).boxed().toArray(Integer[]::new);
        Arrays.sort(array, (i, j) -> nums2[i] - nums2[j]);
        int left = 0, right = n - 1;
        for (int num : nums1) {
            ans[num > nums2[array[left]] ? array[left++] : array[right--]] = num;
        }
        return ans;
    }

    // 盛最多水的容器
    public int maxArea(int[] height) {
        int i = 0, j = height.length - 1, res = 0;
        while (i < j) {
            res = height[i] < height[j] ?
                    Math.max(res, (j - i) * height[i++]) :
                    Math.max(res, (j - i) * height[j--]);
        }
        return res;
    }

    // 统计范围内的元音字符串数
    public int vowelStrings(String[] words, int left, int right) {
        int ans = 0;
        for (int i = left; i <= right; i++) {
            String s = words[i];
            char a = s.charAt(0), b = s.charAt(s.length() - 1);
            if ("aeiou".indexOf(a) != -1 && "aeiou".indexOf(b) != -1) {
                ans++;
            }
        }
        return ans;
    }

    // 超级洗衣机
    public int findMinMoves(int[] machines) {
        int n = machines.length;
        int sum = Arrays.stream(machines).sum();
        if (sum % n != 0) {
            return -1;
        }
        int t = sum / n;
        int ls = 0, rs = sum;
        int ans = 0;
        for (int i = 0; i < n; i++) {
            rs -= machines[i];
            int a = Math.max(t * i - ls, 0);
            int b = Math.max((n - i - 1) * t - rs, 0);
            ans = Math.max(ans, a + b);
            ls += machines[i];
        }
        return ans;
    }

    // 最长平衡子字符串
    public int findTheLongestBalancedSubstring(String s) {
        int n = s.length(), idx = 0, ans = 0;
        while (idx < n) {
            int a = 0, b = 0;
            while (idx < n && s.charAt(idx) == '0' && ++a > 0) {
                idx++;
            }
            while (idx < n && s.charAt(idx) == '1' && ++b > 0) {
                idx++;
            }
            ans = Math.max(ans, Math.min(a, b) * 2);
        }
        return ans;
    }

    // 最多能完成排序的块二
    public int maxChunksToSorted(int[] arr) {
        LinkedList<Integer> stack = new LinkedList<>();
        for (int num : arr) {
            if (!stack.isEmpty() && num < stack.getLast()) {
                int head = stack.removeLast();
                while (!stack.isEmpty() && num < stack.getLast()) {
                    stack.removeLast();
                }
                stack.addLast(head);
            } else {
                stack.addLast(num);
            }
        }
        return stack.size();
    }

    // 咒语和药水的成功对数
    public int[] successfulPairs(int[] spells, int[] potions, long success) {
        int n = spells.length, m = potions.length;
        int[] ans = new int[n];
        Arrays.sort(potions);
        for (int i = 0; i < n; i++) {
            double cur = success * 1.0 / spells[i];
            int l = 0, r = m - 1;
            while (l < r) {
                int mid = l + r >> 1;
                if ((long) potions[mid] * spells[i] >= success) {
                    r = mid;
                } else {
                    l = mid + 1;
                }
            }
            if ((long) potions[r] * spells[i] >= success) {
                ans[i] = m - r;
            }
        }
        return ans;
    }

    // 袋子里最少数目的球
    public int minimumSize(int[] nums, int maxOperations) {
        int l = 1, r = 0x3f3f3f3f;
        while (l < r) {
            int mid = l + r >> 1;
            if (chaeck(nums, mid, maxOperations)) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return r;
    }

    private boolean chaeck(int[] nums, int mid, int maxOperations) {
        int cnt = 0;
        for (int num : nums) {
            cnt += Math.ceil(num * 1.0 / mid) - 1;
        }
        return cnt <= maxOperations;
    }

    // 有界数组中指定下标的最大值
    public int maxValue(int n, int index, int maxSum) {
        int left = 1, right = maxSum;
        while (left < right) {
            int mid = (left + right + 1) >> 1;
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

    // HTML实体解析器
    public String entityParser(String text) {
        Map<String, String> map = new HashMap<>() {{
            put("&quot;", "\"");
            put("&apos", "'");
            put("&amp;", "&");
            put("&gt;", ">");
            put("&lt;", "<");
            put("&frasl;", "/");
        }};
        int n = text.length();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            if (text.charAt(i) == '&') {
                int j = n + 1;
                while (j < n && j - i < 6 && text.charAt(j) != ';') {
                    j++;
                }
                String sub = text.substring(i, Math.min(j + 1, n));
                if (map.containsKey(sub)) {
                    sb.append(map.get(sub));
                    i = j + 1;
                    continue;
                }
            }
            sb.append(text.charAt(i++));
        }
        return sb.toString();
    }

    // 从二叉搜索树到更大和树
    private int s = 0;

    public TreeNode bstToGst(TreeNode root) {
        dfs(root);
        return root;
    }

    private void dfs(TreeNode root) {
        if (root == null) {
            return;
        }
        dfs(root.right);
        s += root.val;
        root.val = s;
        dfs(root.left);
    }

    // 蜡烛之间的盘子
    public int[] platesBetweenCandles(String s, int[][] queries) {
        char[] cs = s.toCharArray();
        int n = cs.length, m = queries.length;
        int[] ans = new int[m], sum = new int[n + 1];
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (cs[i] == '|') {
                list.add(i);
            }
            sum[i + 1] = sum[i] + (cs[i] == '*' ? 1 : 0);
        }
        if (list.size() < 2) {
            return ans;
        }
        for (int i = 0; i < m; i++) {
            int a = queries[i][0], b = queries[i][1];
            int c = -1, d = -1;
            int l = 0, r = list.size() - 1;
            while (l < r) {
                int mid = l + r >> 1;
                if (list.get(mid) > a) {
                    r = mid;
                } else {
                    l = mid + 1;
                }
            }
            if (list.get(r) >= a) {
                c = list.get(r);
            } else {
                continue;
            }
            l = 0;
            r = list.size() - 1;
            while (l < r) {
                int mid = l + r + 1 >> 1;
                if (list.get(mid) < b) {
                    l = mid;
                } else {
                    r = mid - 1;
                }
            }
            if (list.get(r) <= b) {
                d = list.get(r);
            } else {
                continue;
            }
            if (c <= d) {
                ans[i] = sum[d + 1] - sum[c];
            }
        }
        return ans;
    }

    // 雪糕的最大数量
    public int maxIceCream(int[] costs, int coins) {
        int n = costs.length;
        int ans = 0;
        Arrays.sort(costs);
        for (int cost : costs) {
            if (coins >= cost) {
                ans++;
                coins -= cost;
            }
        }
        return ans;
    }

    // 为运算表达式设计优先级
    private char[] cs;

    public List<Integer> diffWaysToCompute(String expression) {
        cs = expression.toCharArray();
        return dfs(0, cs.length - 1);
    }

    private List<Integer> dfs(int l, int r) {
        List<Integer> ans = new ArrayList<>();
        for (int i = l; i <= r; i++) {
            if (cs[i] >= '0' && cs[i] <= '9') {
                continue;
            }
            List<Integer> l1 = dfs(l, i - 1), l2 = dfs(i + 1, r);
            for (int a : l1) {
                for (int b : l2) {
                    int cur = 0;
                    if (cs[i] == '+') {
                        cur = a + b;
                    } else if (cs[i] == '-') {
                        cur = a - b;
                    } else {
                        cur = a * b;
                    }
                    ans.add(cur);
                }
            }
        }
        if (ans.isEmpty()) {
            int cur = 0;
            for (int i = l; i <= r; i++) {
                cur = cur * 10 + (cs[i] - '-');
            }
            ans.add(cur);
        }
        return ans;
    }

    // 重新规划路线
    public int minReorder(int n, int[][] connections) {
        List<int[]>[] e = new List[n];
        for (int i = 0; i < n; i++) {
            e[i] = new ArrayList<>();
        }
        for (int[] edge : connections) {
            e[edge[0]].add(new int[]{edge[1], 1});
            e[edge[1]].add(new int[]{edge[0], 0});
        }
        return dfs(0, -1, e);
    }

    private int dfs(int x, int parent, List<int[]>[] e) {
        int res = 0;
        for (int[] edge : e[x]) {
            if (edge[0] == parent) {
                continue;
            }
            res += edge[1] + dfs(edge[0], x, e);
        }
        return res;
    }

    // 奇偶树
    public boolean isEvenOddTree(TreeNode root) {
        Deque<TreeNode> deque = new ArrayDeque<>();
        boolean flag = true;
        deque.addLast(root);
        while (!deque.isEmpty()) {
            int size = deque.size(), pre = flag ? 0 : 0x3f3f3f3f;
            while (size-- > 0) {
                TreeNode node = deque.pollFirst();
                int cur = node.val;
                if (flag && (cur % 2 == 0 || cur <= pre)) {
                    return false;
                }
                if (!flag && (cur % 2 != 0 || cur >= pre)) {
                    return false;
                }
                pre = cur;
                if (node.left != null) {
                    deque.addLast(node.left);
                }
                if (node.right != null) {
                    deque.addLast(node.right);
                }
            }
            flag = !flag;
        }
        return true;
    }

    // 输出二叉树
    int h, m, n;
    List<List<String>> ans;

    public List<List<String>> printTree(TreeNode root) {
        dfs1(root, 0);
        m = h + 1;
        n = (1 << (h + 1)) - 1;
        ans = new ArrayList<>();
        for (int i = 0; i < m; i++) {
            List<String> cur = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                cur.add("");
            }
            ans.add(cur);
        }
        dfs2(root, 0, (n - 1) / 2);
        return ans;
    }

    private void dfs2(TreeNode root, int x, int y) {
        if (root == null) {
            return;
        }
        ans.get(x).set(y, String.valueOf(root.val));
        dfs2(root.left, x + 1, y - (1 << (h - x - 1)));
        dfs2(root.right, x + 1, y + (1 << (h - x - 1)));
    }

    private void dfs1(TreeNode root, int depth) {
        if (root == null) {
            return;
        }
        h = Math.max(h, depth);
        dfs1(root.left, depth + 1);
        dfs1(root.right, depth + 1);
    }

    // 最大二叉树
    public TreeNode constructMaximumBinaryTree(int[] nums) {
        Deque<TreeNode> deque = new ArrayDeque<>();
        for (int i = 0; i < nums.length; i++) {
            TreeNode node = new TreeNode(nums[i]);
            while (!deque.isEmpty()) {
                TreeNode topNode = deque.peekLast();
                if (topNode.val > node.val) {
                    deque.addLast(node);
                    topNode.right = node;
                    break;
                } else {
                    deque.removeLast();
                    node.left = topNode;
                }
            }
            if (deque.isEmpty()) {
                deque.addLast(node);
            }
        }
        return deque.peek();
    }

    // 下一个更大元素四
    public int[] secondGreaterElement(int[] nums) {
        int[] res = new int[nums.length];
        Arrays.fill(res, -1);
        Deque<Integer> stack = new ArrayDeque<Integer>();
        PriorityQueue<int[]> pq = new PriorityQueue<int[]>((a, b) -> a[0] - b[0]);
        for (int i = 0; i < nums.length; ++i) {
            while (!pq.isEmpty() && pq.peek()[0] < nums[i]) {
                res[pq.poll()[1]] = nums[i];
            }
            while (!stack.isEmpty() && nums[stack.peek()] < nums[i]) {
                pq.offer(new int[]{nums[stack.peek()], stack.peek()});
                stack.pop();
            }
            stack.push(i);
        }
        return res;
    }

    // 字典序最小回文串
    public String makeSmallestPalindrome(String s) {
        char[] cs = s.toCharArray();
        for (int i = 0, j = cs.length - 1; i < j; i++, j--) {
            cs[i] = cs[j] = (char) Math.min(cs[i], cs[j]);
        }
        return new String(cs);
    }


    // 彩灯装饰记录三
    public List<List<Integer>> decorateRecord(TreeNode root) {
        List<List<Integer>> ans = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        if (root != null) {
            queue.add(root);
        }
        while (!queue.isEmpty()) {
            LinkedList<Integer> tmp = new LinkedList<>();
            for (int i = queue.size(); i > 0; i--) {
                TreeNode poll = queue.poll();
                if (ans.size() % 2 == 0) {
                    tmp.addLast(poll.val);
                } else {
                    tmp.addFirst(poll.val);
                }
                if (poll.left != null) {
                    queue.add(poll.left);
                }
                if (poll.right != null) {
                    queue.add(poll.right);
                }
            }
            ans.add(tmp);
        }
        return ans;
    }

    // 寻找峰值一
    public int findPeakElement(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[mid + 1]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    // 寻找峰值二
    public int[] findPeakGrid(int[][] mat) {
        int l = 0, r = mat.length - 1;
        int n = mat[0].length;
        while (l < r) {
            int mid = (l + r) >> 1;
            int j = maxPos(mat[mid]);
            if (mat[mid][j] > mat[mid + 1][j]) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return new int[]{l, maxPos(mat[l])};
    }

    private int maxPos(int[] arr) {
        int j = 0;
        for (int i = 1; i < arr.length; ++i) {
            if (arr[j] < arr[i]) {
                j = i;
            }
        }
        return j;
    }

    // 统计各位数字都不同的数字个数
    public int countNumbersWithUniqueDigits(int n) {
        if (n == 0) {
            return 1;
        }
        int ans = 10;
        for (int i = 2, last = 9; i <= n; i++) {
            int cur = last * (10 - i + 1);
            ans += cur;
            last = cur;
        }
        return ans;
    }

    // 二叉树剪枝
    public TreeNode pruneTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        root.left = pruneTree(root.left);
        root.right = pruneTree(root.right);
        if (root.left != null || root.right != null) {
            return root;
        }
        return root.val == 0 ? null : root;
    }

    // 不浪费原料的汉堡制作方案
    public List<Integer> numOfBurgers(int tomatoSlices, int cheeseSlices) {
        if (tomatoSlices % 2 != 0 || tomatoSlices < 2 * cheeseSlices || cheeseSlices * 4 < tomatoSlices) {
            return new ArrayList<>();
        }
        List<Integer> ans = new ArrayList<>();
        ans.add(tomatoSlices / 2 - cheeseSlices);
        ans.add(cheeseSlices * 2 - tomatoSlices / 2);
        return ans;
    }

    // 笨阶乘
    public int clumsy(int n) {
        Deque<Integer> nums = new ArrayDeque<>();
        Deque<Character> ops = new ArrayDeque<>();
        Map<Character, Integer> map = new HashMap<>() {{
            put('*', 2);
            put('/', 2);
            put('+', 1);
            put('-', 1);
        }};
        char[] cs = new char[]{'*', '/', '+', '-'};
        for (int i = n, j = 0; i > 0; i--, j++) {
            char op = cs[j % 4];
            nums.addLast(i);
            while (!ops.isEmpty() && map.get(ops.peekLast()) >= map.get(op)) {
                deal(nums, ops);
            }
            if (i != 1) {
                ops.add(op);
            }
        }
        while (!ops.isEmpty()) {
            deal(nums, ops);
        }
        return nums.peekLast();
    }

    private void deal(Deque<Integer> nums, Deque<Character> ops) {
        int b = nums.pollLast(), a = nums.pollLast();
        int op = ops.pollLast();
        int ans = 0;
        if (op == '+') {
            ans = a + b;
        } else if (op == '-') {
            ans = a - b;
        } else if (op == '*') {
            ans = a * b;
        } else if (op == '/') {
            ans = a / b;
        }
        nums.addLast(ans);
    }

    // 最简分数
    public List<String> simplifiedFractions(int n) {
        List<String> ans = new ArrayList<>();
        for (int i = 1; i < n; i++) {
            for (int j = i + 1; j <= n; j++) {
                if (gcd(i, j) == 1) {
                    ans.add(i + "/" + j);
                }
            }
        }
        return ans;
    }

    /**
     * 欧几里得方法，求最大公约数GCD
     *
     * @param a
     * @param b
     * @return gcd
     */
    private int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    // Dota2参议院
    public String predictPartyVictory(String senate) {
        int n = senate.length();
        Deque<Integer> radiant = new LinkedList<>();
        Deque<Integer> dire = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            if (senate.charAt(i) == 'R') {
                radiant.offer(i);
            } else {
                dire.offer(i);
            }
        }
        while (!radiant.isEmpty() && !dire.isEmpty()) {
            int ra = radiant.poll(), di = dire.poll();
            if (ra < di) {
                radiant.offer(ra + n);
            } else {
                dire.offer(di + n);
            }
        }
        return !radiant.isEmpty() ? "Radiant" : "Dire";
    }

    // 保龄球游戏的获胜者
    public int isWinner(int[] player1, int[] player2) {
        int a = f(player1), b = f(player2);
        return a > b ? 1 : b > a ? 2 : 0;
    }

    private int f(int[] player) {
        int sum = 0;
        for (int i = 0; i < player.length; i++) {
            int k = (i > 0 && player[i - 1] == 10) || (i > 1 && player[i - 2] == 10) ? 2 : 1;
            sum += k * player[i];
        }
        return sum;
    }


    // 交错字符串
    public boolean isInterleave(String s1, String s2, String s3) {
        int m = s1.length(), n = s2.length();
        if (s3.length() != m + n) {
            return false;
        }
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 1; i <= m && s1.charAt(i - 1) == s3.charAt(i - 1); i++) {
            dp[i][0] = true;
        }
        for (int j = 1; j <= n && s2.charAt(j - 1) == s3.charAt(j - 1); j++) {
            dp[0][j] = true;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i][j] = (dp[i - 1][j] && s3.charAt(i + j - 1) == s1.charAt(i - 1))
                        || (dp[i][j - 1] && s3.charAt(i + j - 1) == s2.charAt(j - 1));
            }
        }
        return dp[m][n];
    }

    // 通过最少操作次数使数组和相等
    public int minOperations(int[] nums1, int[] nums2) {
        if (6 * nums1.length < nums2.length || 6 * nums2.length < nums1.length) {
            return -1;
        }
        int d = Arrays.stream(nums2).sum() - Arrays.stream(nums1).sum();
        if (d < 0) {
            d = -d;
            int[] temp = nums1;
            nums1 = nums2;
            nums2 = temp;
        }
        int[] cnt = new int[6];
        for (int x : nums1) {
            ++cnt[6 - x];
        }
        for (int x : nums2) {
            ++cnt[x - 1];
        }
        for (int i = 5, ans = 0; ; i--) {
            if (i * cnt[i] >= d) {
                return ans + (d + i - 1) / i;
            }
            ans += cnt[i];
            d -= i * cnt[i];
        }
    }

    // 购买两块巧克力
    public int buyChoco(int[] prices, int money) {
        Arrays.sort(prices);
        int res = money - prices[0] - prices[1];
        return res >= 0 ? res : money;
    }

    // 出界的路径数
    int MOD = (int) (1e9 + 7);
    int tm, tn, max;
    int[][][] cache;
    int[][] dirs = new int[][]{{1, 0}, {-1, 0}, {0, -1}, {0, 1}};

    public int findPaths(int _m, int _n, int maxMove, int startRow, int startColumn) {
        tm = _m;
        tn = _n;
        max = maxMove;
        cache = new int[tm][tn][max + 1];
        for (int i = 0; i < tm; i++) {
            for (int j = 0; j < tn; j++) {
                for (int k = 0; k <= max; k++) {
                    cache[i][j][k] = -1;
                }
            }
        }
        return dfs(startRow, startColumn, max);
    }

    private int dfs(int startRow, int startColumn, int max) {
        if (startRow < 0 || startRow >= tm || startColumn < 0 || startColumn >= tn) {
            return 1;
        }
        if (max == 0) {
            return 0;
        }
        if (cache[startRow][startColumn][max] != -1) {
            return cache[startRow][startColumn][max];
        }
        int ans = 0;
        for (int[] d : dirs) {
            int nx = startRow + d[0], ny = startColumn + d[1];
            ans += dfs(nx, ny, max - 1);
            ans %= MOD;
        }
        cache[startRow][startColumn][max] = ans;
        return ans;
    }

    // 一周中的第几天
    static String[] ss = new String[]{"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
    static int[] nums = new int[]{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

    public String dayOfTheWeek(int day, int month, int year) {
        int ans = 4;
        for (int i = 1971; i < year; i++) {
            boolean isLeap = (i % 4 == 0 && i % 100 != 0) || i % 400 == 0;
            ans += isLeap ? 366 : 365;
        }
        for (int i = 1; i < month; i++) {
            ans += nums[i - 1];
            if (i == 2 && ((year % 4 == 0 && year % 100 != 0) || year % 400 == 0)) {
                ans++;
            }
        }
        ans += day;
        return ss[ans % 7];
    }

    // 转换数字的最小运算符
    public int minimumOperations(int[] nums, int start, int goal) {
        Deque<Integer> deque = new ArrayDeque<>();
        Map<Integer, Integer> map = new HashMap<>();
        deque.addLast(start);
        map.put(start, 0);
        while (!deque.isEmpty()) {
            int cur = deque.pollFirst();
            int step = map.get(cur);
            for (int i : nums) {
                int[] result = new int[]{cur + i, cur - i, cur ^ i};
                for (int next : result) {
                    if (next == goal) {
                        return step + 1;
                    }
                    if (next < 0 || next > 1000) {
                        continue;
                    }
                    if (map.containsKey(next)) {
                        continue;
                    }
                    map.put(next, step + 1);
                    deque.addLast(next);
                }
            }
        }
        return -1;
    }

    // 太平洋大西洋水流问题
    int[][] g;

    public List<List<Integer>> pacificAtlantic(int[][] heights) {
        g = heights;
        m = g.length;
        n = g[0].length;
        Deque<int[]> d1 = new ArrayDeque<>(), d2 = new ArrayDeque<>();
        boolean[][] res1 = new boolean[m][n], res2 = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 || j == 0) {
                    res1[i][j] = true;
                    d1.addLast(new int[]{i, j});
                }
                if (i == m - 1 || j == n - 1) {
                    res2[i][j] = true;
                    d2.addLast(new int[]{i, j});
                }
            }
        }
        bfs(d1, res1);
        bfs(d2, res2);
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (res1[i][j] && res2[i][j]) {
                    List<Integer> list = new ArrayList<>();
                    list.add(i);
                    list.add(j);
                    ans.add(list);
                }
            }
        }
        return ans;
    }

    private void bfs(Deque<int[]> d, boolean[][] res) {
        while (!d.isEmpty()) {
            int[] info = d.pollFirst();
            int x = info[0], y = info[1], t = g[x][y];
            for (int[] di : dirs) {
                int nx = x + di[0], ny = y + di[1];
                if (nx < 0 || nx >= m || ny < 0 || ny >= n) {
                    continue;
                }
                if (res[nx][ny] || g[nx][ny] < t) {
                    continue;
                }
                d.addLast(new int[]{nx, ny});
                res[nx][ny] = true;
            }
        }
    }

    // 一年中的第几天
    private static final DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd");

    public int dayOfYear(String date) {
        return LocalDate.parse(date, formatter).getDayOfYear();
    }

    // 经营摩天轮的最大利润
    public int minOperationsMaxProfit(int[] customers, int boardingCost, int runningCost) {
        int ans = -1;
        int mx = 0, cur = 0;
        int wait = 0, i = 0;
        while (i < customers.length || wait > 0) {
            wait += i < customers.length ? customers[i] : 0;
            int up = Math.min(4, wait);
            wait -= up;
            i++;
            cur += up * boardingCost - runningCost;
            if (cur > mx) {
                mx = cur;
                ans = i;
            }
        }
        return ans;
    }

    // 从链表中移除节点
    public ListNode removeNodes(ListNode head) {
        List<Integer> nums = new ArrayList<>();
        while (head != null) {
            nums.add(head.val);
            head = head.next;
        }
        Deque<Integer> deque = new ArrayDeque<>();
        for (int v : nums) {
            while (!deque.isEmpty() && deque.peekLast() < v) {
                deque.pollLast();
            }
            deque.addLast(v);
        }
        ListNode dummy = new ListNode();
        head = dummy;
        while (!deque.isEmpty()) {
            head.next = new ListNode(deque.pollFirst());
            head = head.next;
        }
        return dummy.next;
    }

    // 被列覆盖的最多行数
    public int maximumRows(int[][] matrix, int numSelect) {
        int m = matrix.length, n = matrix[0].length;
        int[] rows = new int[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 1) {
                    rows[i] |= 1 << j;
                }
            }
        }
        int ans = 0;
        for (int mask = 1; mask < 1 << n; mask++) {
            if (Integer.bitCount(mask) != numSelect) {
                continue;
            }
            int t = 0;
            for (int x : rows) {
                if ((x & mask) == x) {
                    t++;
                }
            }
            ans = Math.max(ans, t);
        }
        return ans;
    }

    // 队列中可以看到的人数
    public int[] canSeePersonsCount(int[] heights) {
        int n = heights.length;
        int[] ans = new int[n];
        Deque<Integer> deque = new ArrayDeque<>();
        for (int i = n - 1; i >= 0; i--) {
            while (!deque.isEmpty() && deque.peekLast() < heights[i]) {
                deque.pollLast();
                ans[i]++;
            }
            if (!deque.isEmpty()) {
                ans[i]++;
            }
            deque.offerLast(heights[i]);
        }
        return ans;
    }

    // 在链表中插入最大公约数
    public ListNode insertGreatestCommonDivisors(ListNode head) {
        for (ListNode cur = head; cur.next != null; cur = cur.next.next) {
            cur.next = new ListNode(gcd(cur.val, cur.next.val), cur.next);
        }
        return head;
    }

    // 二叉搜索树中第K小的元素
    int res, k;

    public int kthSmallest(TreeNode root, int k) {
        this.k = k;
        kthSmallestDfs(root);
        return res;
    }

    private void kthSmallestDfs(TreeNode root) {
        if (root == null) {
            return;
        }
        kthSmallestDfs(root.left);
        if (k == 0) {
            return;
        }
        if (--k == 0) {
            res = root.val;
        }
        kthSmallestDfs(root.right);
    }

    // 回旋镖的数量
    public int numberOfBoomerangs(int[][] points) {
        int ans = 0;
        for (int[] p : points) {
            Map<Integer, Integer> cnt = new HashMap<>();
            for (int[] q : points) {
                int dis = (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]);
                cnt.put(dis, cnt.getOrDefault(dis, 0) + 1);
            }
            for (Map.Entry<Integer, Integer> entry : cnt.entrySet()) {
                int m = entry.getValue();
                ans += m * (m - 1);
            }
        }
        return ans;
    }

    // 数组中的第K个最大元素
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int num : nums) {
            if (pq.size() < k) {
                pq.offer(num);
            } else if (num > pq.peek()) {
                pq.poll();
                pq.offer(num);
            }
        }
        return pq.peek();
    }

    // 字符串中的额外字符
    public int minExtraChar(String s, String[] dictionary) {
        Set<String> set = new HashSet<>();
        for (String dic : dictionary) {
            set.add(dic);
        }
        int n = s.length();
        int[] f = new int[n + 1];
        f[0] = 0;
        for (int i = 1; i <= n; i++) {
            f[i] = f[i - 1] + 1;
            for (int j = 0; j < i; j++) {
                if (set.contains(s.substring(j, i))) {
                    f[i] = Math.min(f[i], f[j]);
                }
            }
        }
        return f[n];
    }

    // 删除字串后的字符串最小长度
    public int minLength(String s) {
        Deque<Character> deque = new ArrayDeque<>();
        deque.push(' ');
        for (char c : s.toCharArray()) {
            if ((c == 'B' && deque.peek() == 'A') || (c == 'D' && deque.peek() == 'C')) {
                deque.pop();
            } else {
                deque.push(c);
            }
        }
        return deque.size() - 1;
    }

    // 构造有效字符串的最少插入数
    public int addMinimum(String word) {
        int t = 1;
        char[] charArray = word.toCharArray();
        for (int i = 1; i < charArray.length; i++) {
            if (charArray[i - 1] >= charArray[i]) {
                t++;
            }
        }
        return t * 3 - word.length();
    }

    // 统计出现过一次的公共字符串
    public int countWords(String[] words1, String[] words2) {
        Map<String, Integer> cnt1 = new HashMap<>();
        Map<String, Integer> cnt2 = new HashMap<>();
        for (String str : words1) {
            cnt1.merge(str, 1, Integer::sum);
        }
        for (String str : words2) {
            cnt2.merge(str, 1, Integer::sum);
        }
        int ans = 0;
        for (Map.Entry<String, Integer> entry : cnt1.entrySet()) {
            if (entry.getValue() == 1 && cnt2.getOrDefault(entry.getKey(), 0) == 1) {
                ans++;
            }
        }
        return ans;
    }

    // 构造限制重复的字符串
    public String repeatLimitedString(String s, int repeatLimit) {
        int[] count = new int[26];
        for (int i = 0; i < s.length(); i++) {
            count[s.charAt(i) - 'a']++;
        }
        StringBuilder sb = new StringBuilder();
        int m = 0;
        for (int i = 25, j = 24; i >= 0 && j >= 0; ) {
            if (count[i] == 0) {
                m = 0;
                i--;
            } else if (m < repeatLimit) {
                count[i]--;
                sb.append((char) ('a' + i));
                m++;
            } else if (j >= i || count[j] == 0) {
                j--;
            } else {
                count[j]--;
                sb.append((char) ('a' + j));
                m = 0;
            }
        }
        return sb.toString();
    }

    // 删除排序链表中的重复元素
    public ListNode deleteDuplicates(ListNode head) {
        ListNode cur = head;
        while (cur != null && cur.next != null) {
            if (cur.val == cur.next.val) {
                cur.next = cur.next.next;
            } else {
                cur = cur.next;
            }
        }
        return head;
    }

    // 上下翻转二叉树
    public TreeNode upsideDownBinaryTree(TreeNode root) {
        if (root == null || root.left == null) {
            return root;
        }
        TreeNode left = root.left, right = root.right;
        TreeNode ret = upsideDownBinaryTree(root.left);
        left.left = right;
        left.right = root;
        root.left = root.right = null;
        return ret;
    }

    // 删除排序链表中的重复元素二
    public ListNode deleteDuplicatesTwo(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        if (head.val != head.next.val) {
            head.next = deleteDuplicatesTwo(head.next);
        } else {
            ListNode next = head.next;
            while (next != null && head.val == next.val) {
                next = next.next;
            }
            return deleteDuplicatesTwo(next);
        }
        return head;
    }

    // 最大字符串配对数目
    public int maximumNumberOfStringPairs(String[] words) {
        int ans = 0;
        boolean[][] map = new boolean[26][26];
        for (String s : words) {
            int x = s.charAt(0) - 'a';
            int y = s.charAt(1) - 'a';
            if (map[y][x]) {
                ans++;
            } else {
                map[x][y] = true;
            }
        }
        return ans;
    }

    // 拿出最少数目的魔法豆
    public long minimumRemoval(int[] beans) {
        Arrays.sort(beans);
        long sum = 0, mx = 0;
        int n = beans.length;
        for (int i = 0; i < n; i++) {
            sum += beans[i];
            mx = Math.max(mx, (long) (n - i) * beans[i]);
        }
        return sum - mx;
    }

    // 按分隔符拆分字符串
    public static List<String> splitWordsBySeparator(List<String> words, char separator) {
        List<String> ans = new ArrayList<>();
        for (String str : words) {
            String[] split = str.split(Pattern.quote(String.valueOf(separator)), -1);
            for (String sp : split) {
                if (!sp.isEmpty()) {
                    ans.add(sp);
                }
            }
        }
        return ans;
    }

    // 最大交换
    public int maximumSwap(int num) {
        char[] s = Integer.toString(num).toCharArray();
        int maxIdx = s.length - 1;
        int p = -1, q = 0;
        for (int i = s.length - 2; i >= 0; i--) {
            if (s[i] > s[maxIdx]) {
                maxIdx = i;
            } else if (s[i] < s[maxIdx]) {
                p = i;
                q = maxIdx;
            }
        }
        if (p == -1) {
            return num;
        }
        char temp = s[p];
        s[p] = s[q];
        s[q] = temp;
        return Integer.parseInt(new String(s));
    }

    // 最长交替子数组
    public int alternatingSubarray(int[] nums) {
        int ans = -1;
        int i = 0, n = nums.length;
        while (i < n - 1) {
            if (nums[i + 1] - nums[i] != 1) {
                continue;
            }
            int i0 = i;
            i += 2;
            while (i < n && nums[i] == nums[i - 2]) {
                i++;
            }
            ans = Math.max(ans, i - i0);
            i--;
        }
        return ans;
    }

    // 美丽塔一
    public long maximumSumOfHeights(List<Integer> maxHeights) {
        long ans = 0;
        int n = maxHeights.size();
        for (int i = 0; i < n; i++) {
            int y = maxHeights.get(i);
            long t = y;
            for (int j = i - 1; j >= 0; j--) {
                y = Math.min(y, maxHeights.get(j));
                t += y;
            }
            y = maxHeights.get(i);
            for (int j = i + 1; j < n; j++) {
                y = Math.min(y, maxHeights.get(j));
                t += y;
            }
            ans = Math.max(ans, t);
        }
        return ans;
    }

    // 最大合金数
    public int maxNumberOfAlloys(int n, int k, int budget, List<List<Integer>> composition, List<Integer> stock, List<Integer> cost) {
        int left = 1, right = 200000000, ans = 0;
        while (left <= right) {
            int mid = (left + right) / 2;
            boolean valid = false;
            for (int i = 0; i < k; i++) {
                long spend = 0;
                for (int j = 0; j < n; j++) {
                    spend += Math.max((long) composition.get(i).get(j) * mid - stock.get(j), 0) * cost.get(j);
                }
                if (spend <= budget) {
                    valid = true;
                    break;
                }
            }
            if (valid) {
                ans = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return ans;
    }

    // 水壶问题
    public boolean canMeasureWater(int x, int y, int targetCapacity) {
        Deque<int[]> stack = new LinkedList<>();
        stack.push(new int[]{0, 0});
        Set<Long> seen = new HashSet<>();
        while (!stack.isEmpty()) {
            if (seen.contains(hash(stack.peek()))) {
                stack.pop();
                continue;
            }
            seen.add(hash(stack.peek()));

            int[] state = stack.pop();
            int remain_x = state[0], remain_y = state[1];
            if (remain_x == targetCapacity || remain_y == targetCapacity || remain_x + remain_y == targetCapacity) {
                return true;
            }
            stack.push(new int[]{x, remain_y});
            stack.push(new int[]{remain_x, y});
            stack.push(new int[]{0, remain_y});
            stack.push(new int[]{remain_x, y});
            stack.push(new int[]{remain_x - Math.min(remain_x, y - remain_y), remain_y + Math.min(remain_x, y - remain_y)});
            stack.push(new int[]{remain_x + Math.min(remain_y, x - remain_x), remain_y - Math.min(remain_y, x - remain_x)});
        }
        return false;
    }

    private Long hash(int[] peek) {
        return (long) peek[0] * 1000001 + peek[1];
    }

    // 统计同值子树
    int count = 0;

    public int countUnivalSubtrees(TreeNode root) {
        if (root == null) {
            return 0;
        }
        isUnivalSubtrees(root);
        return count;
    }

    private boolean isUnivalSubtrees(TreeNode root) {
        if (root.left == null && root.right == null) {
            count++;
            return true;
        }
        boolean isUnival = true;
        if (root.left != null) {
            isUnival = isUnivalSubtrees(root.left) && isUnival && root.left.val == root.val;
        }
        if (root.right != null) {
            isUnival = isUnivalSubtrees(root.right) && isUnival && root.right.val == root.val;
        }
        if (isUnival) {
            count++;
            return true;
        } else {
            return false;
        }
    }

    // 使循环数组所有元素相等
    public int minimumSeconds(List<Integer> nums) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        int n = nums.size();
        for (int i = 0; i < n; i++) {
            map.computeIfAbsent(nums.get(i), k -> new ArrayList<>()).add(i);
        }
        int ans = 1 << 30;
        for (List<Integer> idx : map.values()) {
            int m = idx.size();
            int t = idx.get(0) + n - idx.get(m - 1);
            for (int i = 1; i < m; i++) {
                t = Math.max(t, idx.get(i) - idx.get(i - 1));
            }
            ans = Math.min(ans, t / 2);
        }
        return ans;
    }

    // 找出不同元素数目差数组
    public int[] distinctDifferenceArray(int[] nums) {
        int n = nums.length;
        int[] suf = new int[n + 1];
        Set<Integer> set = new HashSet<>();
        for (int i = n - 1; i >= 0; i--) {
            set.add(nums[i]);
            suf[i] = set.size();
        }
        set.clear();
        ;
        int[] ans = new int[n];
        for (int i = 0; i < n; i++) {
            set.add(nums[i]);
            ans[i] = set.size() - suf[i + 1];
        }
        return ans;
    }

    // 至多包含两个不同字符的最长子串
    public int lengthOfLongestSubstringTwoDistinct(String s) {
        int n = s.length();
        if (n < 3) {
            return n;
        }
        int left = 0;
        int right = 0;
        Map<Character, Integer> map = new HashMap<>();
        int ans = 2;
        while (right < n) {
            if (map.size() < 3) {
                map.put(s.charAt(right), right++);
            }
            if (map.size() == 3) {
                Integer min = Collections.min(map.values());
                map.remove(s.charAt(min));
                left = min + 1;
            }
            ans = Math.max(ans, right - left);
        }
        return ans;
    }

    // 石子游戏六
    public int stoneGameVI(int[] aliceValues, int[] bobValues) {
        int n = aliceValues.length;
        Integer[] ids = new Integer[n];
        for (int i = 0; i < n; i++) {
            ids[i] = i;
        }
        Arrays.sort(ids, (i, j) -> aliceValues[j] + bobValues[j] - aliceValues[i] - bobValues[i]);
        int diff = 0;
        for (int i = 0; i < n; i++) {
            diff += i % 2 == 0 ? aliceValues[ids[i]] : -bobValues[ids[i]];
        }
        return Integer.compare(diff, 0);
    }

    // 跳跃游戏六
    public int maxResult(int[] nums, int k) {
        int n = nums.length;
        int[] f = new int[n];
        f[0] = nums[0];
        Deque<Integer> q = new ArrayDeque<>();
        q.add(0);
        for (int i = 1; i < n; i++) {
            if (q.peekFirst() < i - k) {
                q.pollFirst();
            }
            f[i] = f[q.peekFirst()] + nums[i];
            while (!q.isEmpty() && f[i] >= f[q.peekLast()]) {
                q.pollLast();
            }
            q.add(i);
        }
        return f[n - 1];
    }

    // 魔塔游戏
    public int magicTower(int[] nums) {
        long sum = Arrays.stream(nums).sum();
        if (sum < 0) {
            return -1;
        }
        int ans = 0;
        long hp = 1;
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int x : nums) {
            if (x < 0) {
                pq.offer(x);
            }
            hp += x;
            if (hp < 1) {
                hp -= pq.poll();
                ans++;
            }
        }
        return ans;
    }

    // 二叉树的堂兄弟节点二
    public TreeNode replaceValueInTree(TreeNode root) {
        root.val = 0;
        List<TreeNode> nodes = new ArrayList<>();
        nodes.add(root);
        while (!nodes.isEmpty()) {
            List<TreeNode> tmp = new ArrayList<>();
            int s = 0;
            for (TreeNode node : nodes) {
                if (node.left != null) {
                    tmp.add(node.left);
                    s += node.left.val;
                }
                if (node.right != null) {
                    tmp.add(node.right);
                    s += node.right.val;
                }
            }
            for (TreeNode node : nodes) {
                int sub = (node.left == null ? 0 : node.left.val) + (node.right == null ? 0 : node.right.val);
                if (node.left != null) {
                    node.left.val = s - sub;
                }
                if (node.right != null) {
                    node.right.val = s - sub;
                }
            }
            nodes = tmp;
        }
        return root;
    }
}
