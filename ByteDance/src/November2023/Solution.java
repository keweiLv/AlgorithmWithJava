package November2023;

import java.util.*;
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
}
