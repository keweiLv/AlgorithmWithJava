import java.util.HashMap;
import java.util.Map;

/**
 * @Author: Kezi
 * @Date: 2022/4/20 22:45
 */
public class Junier {

    public static void main(String[] args) {
        String s = "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext";
        lengthLongestPath(s);
    }


    public static int lengthLongestPath(String s) {
        Map<Integer, String> map = new HashMap<>();
        int n = s.length();
        String ans = null;
        for (int i = 0; i < n;) {
            int level = 0;
            while (i < n && s.charAt(i) == '\t' && ++level >= 0) {
                i++;
            }
            int j = i;
            boolean isDir = true;
            while (j < n && s.charAt(j) != '\n') {
                if (s.charAt(j++) == '.') {
                    isDir = false;
                }
            }
            String cur = s.substring(i, j);
            String prev = map.getOrDefault(level - 1, null);
            String path = prev == null ? cur : prev + "/" + cur;
            if (isDir) {
                map.put(level, path);
            } else if (ans == null || path.length() > ans.length()) {
                ans = path;
            }
            i = j + 1;
        }
        return ans == null?0:ans.length();
    }
}
