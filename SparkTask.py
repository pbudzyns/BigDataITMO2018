from pyspark.sql import SparkSession
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.functions import collect_set, array_contains
from pyspark.sql.types import ArrayType
import os

os.environ["PYSPARK_PYTHON"] = "/home/pawel/PycharmProjects/HPC/venv/bin/python3.5"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/home/pawel/PycharmProjects/HPC/venv/bin/python3.5"

"""
Process data and build user profile vector with the following characteristics: 
1) count of comments, posts (all), original posts, reposts and likes made by user
2) count of friends, groups, followers
3) count of videos, audios, photos, gifts
4) count of "incoming" (made by other users) comments, max and mean "incoming" comments per post
5) count of "incoming" likes, max and mean "incoming" likes per post
6) count of geo tagged posts
7) count of open / closed (e.g. private) groups a user participates in

Medium:
1) count of reposts from subscribed and not-subscribed groups
2) count of deleted users in friends and followers
3) aggregate (e.g. count, max, mean) characteristics for comments and likes (separtely) made by (a) friends 
   and (b) followers per post
4) aggregate (e.g. count, max, mean) characteristics for comments and likes (separtely) made by (a) friends
   and (b) followers per user
5) find emoji (separately, count of: all, negative, positive, others) in (a) user's posts (b) user's comments  
"""


class SparkTask:

    def __init__(self):
        """
            Configuration for Spark:

            master: address to Master node or local
            path: path to folder with *.parquet files

        """

        self.path = "/home/pawel/bd_parquets/"
        self.master = "local"
        self.app_name = "Spark Task"

        self.spark = SparkSession.builder \
            .master(self.master) \
            .appName(self.app_name) \
            .getOrCreate()

    def read_parquet_file(self, filename):
        return self.spark.read.parquet(self.path + filename)

    def task1a(self):
        """1) count of comments, posts (all), original posts, reposts and likes made by user"""

        user_wall_likes = self.read_parquet_file("userWallLikes.parquet")
        user_wall_posts = self.read_parquet_file("userWallPosts.parquet")
        user_wall_comments = self.read_parquet_file("userWallComments.parquet")

        likes_count = user_wall_likes \
            .groupBy('likerId') \
            .count() \
            .withColumnRenamed('likerId', 'UserId') \
            .withColumnRenamed('count', 'likes')

        posts_count = user_wall_posts \
            .groupBy('from_id') \
            .count() \
            .withColumnRenamed('from_id', 'UserId') \
            .withColumnRenamed('count', 'posts(all)')

        original_posts_count = user_wall_posts \
            .filter(user_wall_posts['is_reposted'] == 'false') \
            .groupBy('from_id') \
            .count() \
            .withColumnRenamed('from_id', 'UserId') \
            .withColumnRenamed('count', 'original_posts')

        reposts_count = user_wall_posts \
            .filter(user_wall_posts['is_reposted'] == 'true') \
            .groupBy('from_id') \
            .count() \
            .withColumnRenamed('from_id', 'UserId') \
            .withColumnRenamed('count', 'reposts')

        comments_cout = user_wall_comments \
            .groupBy('from_id') \
            .count() \
            .withColumnRenamed('from_id', 'UserId') \
            .withColumnRenamed('count', 'comments')

        final_table = comments_cout \
            .join(posts_count, 'UserId') \
            .join(original_posts_count, 'UserId') \
            .join(reposts_count, 'UserId') \
            .join(likes_count, 'UserId')

        return final_table

    def task2a(self):
        """2) count of friends, groups, followers"""

        followers = self.read_parquet_file("followers.parquet")
        friends = self.read_parquet_file("friends.parquet")
        groupsSubs = self.read_parquet_file("userGroupsSubs.parquet")

        friends_count = friends \
            .groupBy('profile') \
            .count() \
            .withColumnRenamed('profile', 'UserId') \
            .withColumnRenamed('count', 'friends')

        groups_count = groupsSubs \
            .groupBy('user') \
            .count() \
            .withColumnRenamed('user', 'UserId') \
            .withColumnRenamed('count', 'groups')

        followers_count = followers \
            .groupBy('profile') \
            .count() \
            .withColumnRenamed('profile', 'UserId') \
            .withColumnRenamed('count', 'followers')

        result_table = friends_count.join(groups_count, 'UserId').join(followers_count, 'UserId')

        return result_table

    def task3a(self):
        """3) count of videos, audios, photos, gifts"""

        friends_profiles = self.read_parquet_file("followerProfiles.parquet")

        result_table = friends_profiles \
            .filter(friends_profiles.counters.isNotNull()) \
            .select(friends_profiles.id.alias("UserId"),
                    friends_profiles.counters.videos.alias("videos"),
                    friends_profiles.counters.audios.alias("audios"),
                    friends_profiles.counters.photos.alias("photos"),
                    friends_profiles.counters.gifts.alias("gifts"))

        return result_table

    def task4a(self):
        """4) count of "incoming" (made by other users) comments, max and mean "incoming" comments per post"""

        user_wall_comments = self.read_parquet_file("userWallComments.parquet")


        incoming_comments_count =user_wall_comments \
            .filter(user_wall_comments['from_id'] != user_wall_comments['post_owner_id']) \
            .groupBy('post_owner_id') \
            .count() \
            .withColumnRenamed('post_owner_id', 'UserId') \
            .withColumnRenamed('count', 'comments_income')


        comments_by_post_count = user_wall_comments \
            .filter(user_wall_comments['from_id'] != user_wall_comments['post_owner_id']) \
            .select('post_id', 'post_owner_id') \
            .groupBy('post_id') \
            .count()

        comment_to_user = user_wall_comments \
            .filter(user_wall_comments['from_id'] != user_wall_comments['post_owner_id']) \
            .select('post_id', 'post_owner_id') \
            .dropDuplicates()

        user_post_comments_summary = comment_to_user \
            .join(comments_by_post_count, 'post_id') \
            .groupBy('post_owner_id') \

        max_comment = user_post_comments_summary \
            .max('count') \
            .withColumnRenamed('post_owner_id', 'UserId') \
            .withColumnRenamed('max(count)', 'max_per_post')

        mean_comment = user_post_comments_summary \
            .mean('count') \
            .withColumnRenamed('post_owner_id', 'UserId') \
            .withColumnRenamed('avg(count)', 'mean_per_post')

        result_table = incoming_comments_count \
            .join(max_comment, 'UserId') \
            .join(mean_comment, 'UserId')

        return result_table

    def task5a(self):
        """5) count of "incoming" likes, max and mean "incoming" likes per post"""

        userWallLikes = self.read_parquet_file("userWallLikes.parquet")

        likes_per_post = userWallLikes \
            .filter(userWallLikes['ownerId'] != userWallLikes['likerId']) \
            .groupBy('itemId') \
            .count()

        post_to_user = userWallLikes \
            .filter(userWallLikes['ownerId'] != userWallLikes['likerId']) \
            .select('itemId', 'ownerId') \
            .dropDuplicates()

        user_likes_summary = post_to_user.join(likes_per_post, 'itemId').groupBy('ownerId')

        total_likes_per_post = user_likes_summary \
            .sum('count') \
            .withColumnRenamed('ownerId', 'UserId') \
            .withColumnRenamed('sum(count)', 'total_likes')

        max_likes_per_post = user_likes_summary \
            .max('count') \
            .withColumnRenamed('ownerId', 'UserId') \
            .withColumnRenamed('max(count)', 'max_per_post')

        mean_likes_per_post = user_likes_summary \
            .mean('count') \
            .withColumnRenamed('ownerId', 'UserId') \
            .withColumnRenamed('avg(count)', 'mean_per_post')

        result_table = total_likes_per_post \
            .join(max_likes_per_post, 'UserId') \
            .join(mean_likes_per_post, 'UserId')

        return result_table

    def task6a(self):
        """6) count of geo tagged posts"""

        userWallPosts = self.read_parquet_file("userWallPosts.parquet")

        geo_tagged_posts_count = userWallPosts \
            .filter(userWallPosts['geo.coordinates'] != 'null') \
            .groupBy('owner_id') \
            .count() \
            .withColumnRenamed('owned_id', 'UserId') \
            .withColumnRenamed('count', 'geo_tagged_posts') \

        result_table = geo_tagged_posts_count

        return result_table

    def task7a(self):
        """7) count of open / closed (e.g. private) groups a user participates in"""

        groupsProfiles = self.read_parquet_file("groupsProfiles.parquet")
        userGroupsSubs = self.read_parquet_file("userGroupsSubs.parquet")

        invert_id = UserDefinedFunction(lambda x: -int(x))
        user_to_group = userGroupsSubs \
            .select("user", invert_id("group")) \
            .withColumnRenamed("<lambda>(group)", "group")\
            .dropDuplicates()

        group_type = groupsProfiles\
            .select("id", "is_closed")\
            .withColumnRenamed("id", "group")\
            .dropDuplicates()

        user_to_group_type = user_to_group\
            .join(group_type, "group")\

        opened_groups = user_to_group_type\
            .filter(user_to_group_type['is_closed'] == 0)\
            .groupBy("user")\
            .count()\
            .withColumnRenamed("count", "opened")

        closed_groups = user_to_group_type\
            .filter(user_to_group_type['is_closed'] > 0)\
            .groupBy("user")\
            .count()\
            .withColumnRenamed("count", "closed")

        result_table = opened_groups.join(closed_groups, "user")

        return result_table

    def task1b(self):
        """1) count of reposts from subscribed and not-subscribed groups"""

        userWallPosts = self.read_parquet_file("userWallPosts.parquet")
        userGroupsSubs = self.read_parquet_file("userGroupsSubs.parquet")

        reposts_t = userWallPosts \
            .filter(userWallPosts.is_reposted) \
            .select('owner_id', 'repost_info.orig_owner_id')\
            .withColumnRenamed("owner_id", "user")

        reposts = reposts_t.filter(reposts_t["orig_owner_id"] < 0)

        user_to_group_sub = userGroupsSubs\
            .select("user", "group")\
            .groupBy("user")\
            .agg(collect_set("group"))\
            .withColumnRenamed("collect_set(group)", "groups")

        def contains(id, groups):
            if not groups:
                return False
            if str(id) in groups:
                return True
            else:
                return False

        contains_udf = UserDefinedFunction(contains)

        temp = reposts.join(user_to_group_sub, "user", how="left_outer")

        reposts_from = temp\
            .withColumn("from_subscribed", contains_udf(temp.orig_owner_id, temp.groups))

        reposts_from_subscribed = reposts_from\
            .filter(reposts_from.from_subscribed == 'true')\
            .select('user')\
            .groupBy('user')\
            .count()\
            .withColumnRenamed("count", "from_subscribed")

        reposts_not_from_subscribed = reposts_from \
            .filter(reposts_from['from_subscribed'] == 'false') \
            .select('user')\
            .groupBy("user")\
            .count()\
            .withColumnRenamed("count", "not_from_subscribed")

        result_table = reposts_from_subscribed.join(reposts_not_from_subscribed, 'user')

        return result_table




if __name__ == "__main__":
    spark = SparkTask()
    # spark.task1a().show()
    # spark.task2a().show()
    # spark.task3a().show()
    # spark.task4a().show()
    # spark.task5a().show()
    # spark.task6a().show()
    # spark.task7a().show()
    # spark.task1b().show()
    # print(res.show())
