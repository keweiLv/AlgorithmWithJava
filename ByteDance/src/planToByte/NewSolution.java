package planToByte;

import java.util.*;

/**
 * @author Kezi
 * @date 2023年02月03日 23:07
 */
public class NewSolution {

    // 二叉树着色游戏
    private int x, lsz, rsz;

    public boolean btreeGameWinningMove(TreeNode root, int n, int x) {
        this.x = x;
        dfs(root);
        return Math.max(Math.max(lsz, rsz), n - 1 - lsz - rsz) * 2 > n;
    }

    private int dfs(TreeNode node) {
        if (node == null) {
            return 0;
        }
        int ls = dfs(node.left);
        int rs = dfs(node.right);
        if (node.val == x) {
            lsz = ls;
            rsz = rs;
        }
        return ls + rs + 1;
    }

    // 你能构造出连续值的最大数目
    public int getMaximumConsecutive(int[] coins) {
        int m = 0;
        Arrays.sort(coins);
        for (int coin : coins) {
            if (coin > m + 1) {
                break;
            }
            m += coin;
        }
        return m + 1;
    }

    // 最长连续序列
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        int longestLen = 0;
        for (int num : set) {
            if (!set.contains(num - 1)) {
                int curNum = num + 1;
                int curLen = 1;
                while (set.contains(curNum + 1)) {
                    curLen++;
                    curNum++;
                }
                longestLen = Math.max(longestLen, curLen);
            }
        }
        return longestLen;
    }

    // 有效的括号
    private static final Map<Character, Character> map = new HashMap<Character, Character>() {{
        put('{', '}');
        put('[', ']');
        put('(', ')');
        put('?', '?');
    }};

    public boolean isValid(String s) {
        if (s.length() > 0 && !map.containsKey(s.charAt(0))) {
            return false;
        }
        LinkedList<Character> stack = new LinkedList<Character>() {{
            add('?');
        }};
        for (Character c : s.toCharArray()) {
            if (map.containsKey(c)) {
                stack.addLast(c);
            } else if (!map.get(stack.removeLast()).equals(c)) {
                return false;
            }
        }
        return stack.size() == 1;
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

    // 计算布尔二叉树的值
    public boolean evaluateTree(TreeNode root) {
        if (root.left == null) {
            return root.val == 1;
        }
        boolean l = evaluateTree(root.left);
        boolean r = evaluateTree(root.right);
        return root.val == 2 ? l || r : l && r;
    }

    // 警告一小时内使用相同员工卡大于等于三次的人
    public List<String> alertNames(String[] keyName, String[] keyTime) {
        Map<String, List<Integer>> timeMap = new HashMap<>();
        for (int i = 0; i < keyName.length; i++) {
            String name = keyName[i];
            String time = keyTime[i];
            timeMap.putIfAbsent(name, new ArrayList<>());
            int cnt = Integer.parseInt(time.substring(0, 2)) * 60 + Integer.parseInt(time.substring(3));
            timeMap.get(name).add(cnt);
        }
        List<String> res = new ArrayList<>();
        Set<String> strings = timeMap.keySet();
        for (String name : strings) {
            List<Integer> list = timeMap.get(name);
            Collections.sort(list);
            for (int i = 0; i < list.size(); i++) {
                int time1 = list.get(i - 2), time2 = list.get(i);
                int diff = time2 - time1;
                if (diff > 60) {
                    res.add(name);
                    break;
                }
            }
        }
        Collections.sort(res);
        return res;
    }
}